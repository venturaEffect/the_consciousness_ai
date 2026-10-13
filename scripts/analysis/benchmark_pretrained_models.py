"""Cost of the optional pretrained models on this machine, and one content check for vision.

The training loop loads none of these models. This script measures what each one would
cost if its optional module were used, and it changes no model id in the code.

COST, per model, device and precision. Parameters, time to load, time per call (a loop of
at least 3 calls and at least `--seconds` seconds, after 2 warm-up calls), the largest GPU
memory torch allocated, the growth of process memory, and the mean GPU board power from
`nvidia-smi` during the loop. A value that cannot be read is None. A model that fails to
load or run is reported with its error text and no numbers.
    vision     one image, 224 px (384 px for the PE model), one forward pass
    text       one prompt, 32 new tokens, greedy
    vlm        one 224 px image with a prompt, 32 new tokens, greedy
    embedding  one sentence

CONTENT CHECK FOR VISION. 280 DMTS trials (14 episodes of 20 trials, seed 42, the sample
frame of each trial). The frame goes through the encoder, the patch grid is averaged to
1x1 and to 4x4 cells, and a linear probe (StandardScaler + LogisticRegression, as
`probe_perception_decodability.linear_decode`) reads shape and colour. Whole stimuli are
held out: 5 splits, 30 percent of the stimuli in the test side, seeds 0 to 4. Chance is
1/6. A baseline uses the pixels of the frame, averaged to a 16x16 grid.

PREDICTIONS for the content check, written 2026-10-13 before any model was run.
  V1. For `facebook/dinov2-base` and `facebook/dinov2-with-registers-small`, shape from
      the 4x4 grid is above 0.50 at 5 of 5 splits.
  V2. For the same two models, colour from the 4x4 grid is above 0.50 at 5 of 5 splits.
No prediction is made for the PE model or for the pixel baseline. A false prediction is
reported as FAILED. For reference, the trained tectum of this project reads shape of an
unseen stimulus at chance (`docs/results/thermodynamic_survival_2026_10.md`). That
number comes from another probe and another split, so it is not compared directly.

This is a cost study and a decode on a closed task. It is not evidence about
consciousness, and it moves no indicator.

Usage:
    python -m scripts.analysis.benchmark_pretrained_models --output-json runs/model_benchmark/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

VISION_MODELS = ("facebook/dinov2-base", "facebook/dinov2-with-registers-small", "timm/vit_pe_core_tiny_patch16_384.fb")
TEXT_MODELS = ("Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3.5-0.8B")
VLM_MODELS = ("Qwen/Qwen3.5-0.8B", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
EMBEDDING_MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "minishlab/potion-base-8M", "ibm-granite/granite-embedding-30m-english")
PROMPT = "I am in a dark room. I hear a sound on my left. Describe what I do next."
SENTENCE = "The light was out of view and a sound came from the left."
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NEW_TOKENS, SPLITS, TEST_SHARE = 32, 5, 0.3


# ----------------------------------------------------------------------------- measurement helpers


def count_parameters(module: torch.nn.Module) -> int:
    """Number of parameter values, each shared tensor counted once."""
    return sum(p.numel() for p in {id(p): p for p in module.parameters()}.values())


def summarise_times(times: Sequence[float]) -> Dict[str, Optional[float]]:
    if not times:
        return {"runs": 0, "min_s": None, "median_s": None, "max_s": None}
    return {"runs": len(times), "min_s": min(times), "median_s": statistics.median(times), "max_s": max(times)}


def timed_loop(call: Callable[[], object], seconds: float, minimum_runs: int) -> List[float]:
    """Time of each call, repeated until both the time and the count are reached."""
    times, started = [], time.perf_counter()
    while len(times) < minimum_runs or time.perf_counter() - started < seconds:
        before = time.perf_counter()
        call()
        times.append(time.perf_counter() - before)
    return times


def read_gpu_power() -> Optional[float]:
    """GPU board power in watts from nvidia-smi, or None when it cannot be read."""
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], timeout=10)
        return float(text.decode().strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def mean_power(samples: Sequence[Optional[float]]) -> Optional[float]:
    present = [s for s in samples if s is not None]
    return sum(present) / len(present) if present else None


class PowerSampler:
    """Reads the GPU power about twice per second in a thread, between start and stop."""

    def __init__(self) -> None:
        self.samples: List[Optional[float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(read_gpu_power())
            self._stop.wait(0.4)

    def __enter__(self) -> "PowerSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure(load: Callable[[str, torch.dtype], Tuple[object, Callable[[], object]]], device: str, dtype: torch.dtype,
            seconds: float) -> Dict[str, object]:
    """Load one model and time its call. Numbers only for what ran; an error is returned as text."""
    import psutil
    process = psutil.Process()
    rss_before = process.memory_info().rss
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, call = load(device, dtype)
    load_seconds = time.perf_counter() - started

    def synced_call() -> None:
        call()
        _sync(device)

    with torch.inference_mode():
        timed_loop(synced_call, seconds=0.0, minimum_runs=2)
        with PowerSampler() as sampler:
            times = timed_loop(synced_call, seconds=seconds, minimum_runs=3)
    row = {"parameters": count_parameters(model) if isinstance(model, torch.nn.Module) else None, "load_s": load_seconds,
           "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None,
           "rss_growth_mb": (process.memory_info().rss - rss_before) / 1e6,
           "power_w": mean_power(sampler.samples), "power_samples": len(sampler.samples)}
    row.update(summarise_times(times))
    return row


# ------------------------------------------------------------------------------------ model loaders


def load_vision(model_id: str) -> Callable:
    def load(device: str, dtype: torch.dtype):
        if model_id.startswith("timm/"):
            import timm
            model = timm.create_model("hf_hub:" + model_id, pretrained=True).to(device=device, dtype=dtype).eval()
            size = model.pretrained_cfg["input_size"][-1]
            image = torch.rand(1, 3, size, size, device=device, dtype=dtype)
            return model, lambda: model.forward_features(image)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
        image = torch.rand(1, 3, 224, 224, device=device, dtype=dtype)
        return model, lambda: model(pixel_values=image)
    return load


def _generation_model(model_id: str):
    """The causal language model class, or the image-text class for models that have only that."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        return AutoModelForCausalLM.from_pretrained(model_id)
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(model_id)


def load_text(model_id: str) -> Callable:
    def load(device: str, dtype: torch.dtype):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = _generation_model(model_id).to(device=device, dtype=dtype).eval()
        inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
        return model, lambda: model.generate(**inputs, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False)
    return load


def load_vlm(model_id: str) -> Callable:
    def load(device: str, dtype: torch.dtype):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
        image = Image.fromarray((np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(device)
        inputs = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        return model, lambda: model.generate(**inputs, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False)
    return load


def load_embedding(model_id: str) -> Callable:
    def load(device: str, dtype: torch.dtype):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id, device=device)
        if dtype == torch.float16:
            model.half()
        return model, lambda: model.encode([SENTENCE])
    return load


ROLES = {"vision": (VISION_MODELS, load_vision), "text": (TEXT_MODELS, load_text),
         "vlm": (VLM_MODELS, load_vlm), "embedding": (EMBEDDING_MODELS, load_embedding)}
SETTINGS = (("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.float16))


def cost_rows(roles: Sequence[str], seconds: float) -> List[Dict[str, object]]:
    rows = []
    for role in roles:
        models, loader = ROLES[role]
        for model_id in models:
            for device, dtype in SETTINGS:
                row = {"role": role, "model": model_id, "device": device, "precision": str(dtype).replace("torch.", "")}
                try:
                    row.update(measure(loader(model_id), device, dtype, seconds))
                except Exception as error:  # the error text is the record, and no number is written for this row
                    row["error"] = "%s: %s" % (type(error).__name__, " ".join(str(error).split())[:300])
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


# ----------------------------------------------------------------------------- vision content check


def pool_patch_grid(grid: np.ndarray, cells: int) -> np.ndarray:
    """Average a patch grid [rows, H, W, channels] into cells x cells blocks, flattened per row."""
    rows, height, width, channels = grid.shape
    if height % cells or width % cells:
        raise ValueError("a %dx%d grid does not divide into %d cells per side" % (height, width, cells))
    blocks = grid.reshape(rows, cells, height // cells, cells, width // cells, channels).mean(axis=(2, 4))
    return blocks.reshape(rows, cells * cells * channels)


def stimulus_splits(stimulus: np.ndarray, splits: int):
    """Train and test rows with whole stimuli held out, one pair per split."""
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=splits, test_size=TEST_SHARE, random_state=0)
    return list(splitter.split(np.zeros(len(stimulus)), groups=stimulus))


def unseen_stimulus_accuracy(features: np.ndarray, labels: np.ndarray, stimulus: np.ndarray, splits: int) -> List[float]:
    """Accuracy of the linear probe on held-out stimuli, one value per split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    scores = []
    for train, test in stimulus_splits(stimulus, splits):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        probe.fit(features[train], labels[train])
        scores.append(float(probe.score(features[test], labels[test])))
    return scores


def collect_dmts_frames(episodes: int = 14, seed: int = 42) -> Dict[str, np.ndarray]:
    """The sample frame and the labels of every DMTS trial, with the trial loop of the project's probes."""
    from simulations.environments.dmts_env import DMTSEnv
    env, rng = DMTSEnv(num_trials=20), np.random.default_rng(seed)
    frames, labels = [], {"shape": [], "color": [], "size": []}
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        current, done, steps = None, False, 0
        while not done and steps < 4000:
            phase = info.get("phase")
            if phase == "sample":
                current = (obs.copy(), info["sample_shape"], info["sample_color"], info["sample_size"])
            elif phase == "choice" and current is not None:
                frames.append(current[0])
                for name, item in zip(("shape", "color", "size"), current[1:]):
                    labels[name].append(item)
                current = None
            action = 0 if phase != "choice" else int(rng.integers(1, env.num_choices + 1))
            obs, _, terminated, truncated, info = env.step(action)
            done, steps = terminated or truncated, steps + 1
    return {"frames": np.stack(frames), **{k: np.array(v) for k, v in labels.items()}}


def _prepare(frames: np.ndarray, size: int) -> torch.Tensor:
    images = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def patch_grid(model_id: str, frames: np.ndarray, device: str) -> np.ndarray:
    """Patch tokens of every frame as [rows, H, W, channels], without class and register tokens."""
    if model_id.startswith("timm/"):
        import timm
        model = timm.create_model("hf_hub:" + model_id, pretrained=True).to(device).eval()
        size, prefix = model.pretrained_cfg["input_size"][-1], model.num_prefix_tokens
        forward = model.forward_features
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        size, prefix = 224, 1 + int(getattr(model.config, "num_register_tokens", 0))
        forward = lambda images: model(pixel_values=images).last_hidden_state  # noqa: E731
    tokens = []
    with torch.inference_mode():
        for batch in torch.split(_prepare(frames, size), 16):
            tokens.append(forward(batch.to(device))[:, prefix:, :].float().cpu().numpy())
    tokens = np.concatenate(tokens)
    side = int(round(tokens.shape[1] ** 0.5))
    return tokens.reshape(len(tokens), side, side, tokens.shape[2])


def pixel_grid(frames: np.ndarray) -> np.ndarray:
    images = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    return torch.nn.functional.adaptive_avg_pool2d(images, 16).permute(0, 2, 3, 1).numpy()


def decode_rows(device: str) -> List[Dict[str, object]]:
    records = collect_dmts_frames()
    stimulus = np.array(["%s|%s|%s" % item for item in zip(records["shape"], records["color"], records["size"])])
    grids = {"pixels 16x16": lambda: pixel_grid(records["frames"])}
    grids.update({model_id: (lambda model_id=model_id: patch_grid(model_id, records["frames"], device)) for model_id in VISION_MODELS})
    rows = []
    for name, make_grid in grids.items():
        grid = make_grid()
        for cells in (1, 4):
            features = pool_patch_grid(grid, cells).astype(np.float64)
            for label in ("shape", "color"):
                scores = unseen_stimulus_accuracy(features, records[label], stimulus, SPLITS)
                rows.append({"features": name, "cells": cells, "label": label, "dims": features.shape[1],
                             "scores": scores, "trials": len(stimulus), "stimuli": len(set(stimulus))})
                print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roles", nargs="*", default=list(ROLES), choices=list(ROLES))
    parser.add_argument("--seconds", type=float, default=8.0, help="least time of each timed loop")
    parser.add_argument("--skip-decode", action="store_true", help="leave out the content check for vision")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "idle_power_w": mean_power([read_gpu_power() for _ in range(5)]),
              "cost": cost_rows(args.roles, args.seconds),
              "decode": [] if args.skip_decode else decode_rows("cuda" if torch.cuda.is_available() else "cpu")}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
