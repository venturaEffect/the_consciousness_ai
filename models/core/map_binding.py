"""Stage 3 of the binding design: one oscillator for each place on the tectum map.

`docs/decisions/2026_10_14_binding_stage3_map_oscillators.md`, design A with the three
corrections of its review.

- Units. One unit vector of DIMENSIONS components for each cell of the map. AKOrN reports
  that 2 components underfit and uses 4. On a 2-D lattice, 2-component phases can keep
  vortices that block synchrony; with more components they cannot form stable point
  defects.
- Drive. The visual drive of a cell is its feature norm, scaled between the median and the
  largest norm of the map. The audio drive is a Gaussian area around the cell of the sound
  (sigma 2 cells, so about 3 to 5 cells wide), because auditory fields are 2 to 4 times
  wider than visual ones. The two combine as either one: 1 - (1 - v)(1 - a).
- Coupling. Each active cell turns toward the drive-weighted mean of its 8 neighbours
  (local cooperation) and away from the drive-weighted mean of the active cells that are
  not its neighbours (one global inhibitor, the Imc of the midbrain). Inactive cells do
  not move. After Terman and Wang (1995).
- Time. ITERATIONS updates run inside each environment step. A phase is in cycles per
  step, with no Hz reading. A cell whose drive rises from below ACTIVE restarts with a
  random phase, so a new stimulus starts a new rhythm.
- Read-out. The selected group is the set of active cells that are synchronised (cosine
  of at least GROUP_COSINE) with one active cell, chosen so that the group holds the most
  drive. A sense gets the binding factor 1 + 0.5 * (share of its drive inside the group),
  the same range as the workspace layer.

ITERATIONS, DT, INHIBITION, ACTIVE, GROUP_COSINE and AUDIO_SIGMA are assumed values. They
are not fitted to data.

MEASURED LIMIT OF THIS PHASE VERSION, 2026-10-14 (unit-level runs, not an agent run). With
an inhibitor of 0.5, one 6 by 6 object did not synchronise at all (coherence 0.011),
because the inhibitor also acts between cells of the same object. With 0.02, a 12 by 12
object reached a coherence of 0.96 to 0.99 and two objects separated (-0.98 to -1.00)
only after 100 steps of 10 updates (3 seeds). The agent moves about 1.7 cells per step,
so this is too slow for a moving object. LEGION avoids this with relaxation oscillators,
in which activity jumps through a whole connected region within one cycle (Wang and
Terman, NIPS 1994, read in full). The faithful version is the next step.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

GRID, DIMENSIONS = 16, 4
ITERATIONS, DT, INHIBITION = 10, 0.2, 0.02
ACTIVE, GROUP_COSINE, AUDIO_SIGMA = 0.05, 0.9, 2.0

_NEIGHBOURS = torch.ones(1, 1, 3, 3)
_NEIGHBOURS[0, 0, 1, 1] = 0.0


def visual_drive(vision_grid: torch.Tensor) -> torch.Tensor:
    """[H, W] in [0, 1]: (feature norm - median) / (largest - median), cut. Zero for a flat map."""
    # The map works on the CPU (scipy groups it), whatever the device of the encoder.
    norms = vision_grid[0].detach().float().cpu().norm(dim=0)
    top, middle = norms.max(), norms.median()
    if float(top - middle) <= 0.0:
        return torch.zeros_like(norms)
    return torch.clamp((norms - middle) / (top - middle), 0.0, 1.0)


def audio_drive(audio_spatial: torch.Tensor, height: int, width: int, sigma: float = AUDIO_SIGMA) -> torch.Tensor:
    """[H, W]: a Gaussian area around the cell of the sound, placed as TopographicMap places it."""
    if not bool(torch.any(audio_spatial != 0)):
        return torch.zeros(height, width)
    ax, ay = audio_spatial[0, 0, 0].item(), audio_spatial[0, 0, 1].item()
    gx = int(torch.clamp(torch.tensor((ax + 1) / 2 * (width - 1)), 0, width - 1).int())
    gy = int(torch.clamp(torch.tensor((ay + 1) / 2 * (height - 1)), 0, height - 1).int())
    rows = torch.arange(height, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(width, dtype=torch.float32).view(1, -1)
    return torch.exp(-((cols - gx) ** 2 + (rows - gy) ** 2) / (2.0 * sigma ** 2))


def combine(vision: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Either one: 1 - (1 - v)(1 - a)."""
    return 1.0 - (1.0 - vision) * (1.0 - audio)


def _neighbour_sum(field: torch.Tensor) -> torch.Tensor:
    """Sum over the 8 neighbours of each cell, for a [H, W] or [H, W, D] field."""
    if field.dim() == 2:
        return F.conv2d(field[None, None], _NEIGHBOURS, padding=1)[0, 0]
    stacked = field.permute(2, 0, 1)[:, None]
    return F.conv2d(stacked, _NEIGHBOURS, padding=1)[:, 0].permute(1, 2, 0)


class MapOscillators:
    """The oscillator map. `step` once for each environment step, then `selected_group` and `factors`."""

    def __init__(self, size: int = GRID, seed: int = 0, iterations: int = ITERATIONS,
                 inhibition: float = INHIBITION):
        self.generator = torch.Generator().manual_seed(int(seed))
        self.iterations, self.inhibition = iterations, inhibition
        self.phases = self._random(size, size)
        self.previous_drive = torch.zeros(size, size)

    def _random(self, *shape) -> torch.Tensor:
        return F.normalize(torch.randn(*shape, DIMENSIONS, generator=self.generator), dim=-1)

    def step(self, drive: torch.Tensor) -> None:
        """Restart the cells that just became active, then run the updates of one step."""
        drive = drive.detach().float()
        rising = (drive >= ACTIVE) & (self.previous_drive < ACTIVE)
        if bool(rising.any()):
            self.phases[rising] = self._random(int(rising.sum()))
        for _ in range(self.iterations):
            self.phases = self._update(self.phases, drive)
        self.previous_drive = drive.clone()

    def _update(self, phases: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        weighted = drive[..., None] * phases
        local, local_mass = _neighbour_sum(weighted), _neighbour_sum(drive)
        far = weighted.sum(dim=(0, 1)) - local - weighted
        far_mass = (drive.sum() - local_mass - drive).clamp_min(0.0)
        pull = local / local_mass.clamp_min(1e-8)[..., None]
        pull = pull - self.inhibition * far / far_mass.clamp_min(1e-8)[..., None]
        tangent = pull - (pull * phases).sum(-1, keepdim=True) * phases
        return F.normalize(phases + DT * drive[..., None] * tangent, dim=-1)

    def selected_group(self, drive: torch.Tensor) -> torch.Tensor:
        """[H, W] bool: the synchronised group of active cells that holds the most drive."""
        active = drive >= ACTIVE
        if not bool(active.any()):
            return torch.zeros_like(active)
        cells = self.phases[active]
        together = (cells @ cells.T) >= GROUP_COSINE
        best = int(torch.argmax(together.float() @ drive[active]))
        group = torch.zeros_like(active)
        group[active] = together[best]
        return group

    @staticmethod
    def factors(group: torch.Tensor, vision: torch.Tensor, audio: torch.Tensor) -> dict:
        """{"vision": f, "audio": f} with f = 1 + 0.5 * (share of the drive of that sense inside the group)."""
        result = {}
        for name, drive in (("vision", vision), ("audio", audio)):
            total = float(drive.sum())
            share = float(drive[group].sum()) / total if total > 0.0 else 0.0
            result[name] = 1.0 + 0.5 * share
        return result


def select_group(labels, vision: torch.Tensor, audio: torch.Tensor) -> tuple:
    """(factors, record) for groups given as labels (0 is no group): the group with the most combined drive."""
    from scipy import ndimage
    count = int(labels.max())
    if count == 0:
        return {"vision": 1.0, "audio": 1.0}, {"blocks": 0, "group": []}
    masses = ndimage.sum(combine(vision, audio).numpy(), labels, index=range(1, count + 1))
    group = torch.from_numpy(labels == int(masses.argmax()) + 1)
    cells = [int(i) for i in torch.nonzero(group.flatten()).flatten()]
    return MapOscillators.factors(group, vision, audio), {"blocks": count, "group": cells}


class LegionLimit:
    """Option B of stage 3: the end result that Terman and Wang prove for LEGION.

    Wang and Terman (NIPS 1994, read in full; Terman and Wang, Physica D 1995) show for
    their network of relaxation oscillators, with 4 nearest neighbours and one global
    inhibitor, in the limit of a small epsilon: all oscillators of one connected
    stimulated area jump up together, different areas fire at different times, and at
    most one area is active at a time. A neighbour link exists only between two
    stimulated cells (their dynamic normalisation for binary input). This class computes
    those groups directly and does not simulate the oscillation. A cell is stimulated when
    its combined drive is at least `threshold`. The group with the most drive is selected,
    as the midbrain selects one place (Imc, Knudsen 2018). Option A, the full equations,
    is a later check of this result.

    MEASURED FAILURE of the defaults (feature norm, threshold ACTIVE), 2026-10-14: one
    group at 0.951 to 0.994 of the steps in three runs, so nothing was selected
    (`docs/results/binding_map_stage3_2026_10.md`).
    """

    def __init__(self, threshold: float = ACTIVE, drive=None):
        self.threshold = threshold
        self.drive = drive or visual_drive

    def observe(self, vision_grid: torch.Tensor, audio_spatial: torch.Tensor) -> tuple:
        """(factors, record) from the encoder map and the sound direction."""
        height, width = vision_grid.shape[-2], vision_grid.shape[-1]
        return self.update(self.drive(vision_grid), audio_drive(audio_spatial, height, width))

    def update(self, vision: torch.Tensor, audio: torch.Tensor) -> tuple:
        """(factors, record) for one step. record holds the number of groups and the cells of the selected one."""
        from scipy import ndimage
        labels, _ = ndimage.label((combine(vision, audio) >= self.threshold).numpy())
        return select_group(labels, vision, audio)


# Candidate drives of `docs/decisions/2026_10_14_map_binding_drive.md`. CENTRE_SIGMA is
# assumed (one cell, the spacing of the map). SURROUND_RATIO is the surround to centre
# ratio of the salience model for luminance and colour in White and colleagues (Nat.
# Commun. 2017), whose output predicted the visual neurons of the superficial colliculus.
CENTRE_SIGMA, SURROUND_RATIO, HALF_MAXIMUM = 1.0, 6.7, 0.5


@functools.lru_cache(maxsize=8)
def _gaussian_mean(height: int, width: int, sigma: float) -> torch.Tensor:
    """[H*W, H*W]: row i holds the weights of the Gaussian mean at cell i over the map (rows sum to 1)."""
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack([rows.flatten(), cols.flatten()], dim=1).double()
    weights = torch.exp(-torch.cdist(points, points) ** 2 / (2.0 * sigma ** 2))
    return weights / weights.sum(dim=1, keepdim=True)


def salience_drive(vision_grid: torch.Tensor) -> torch.Tensor:
    """[H, W] in [0, 1]: the norm of centre minus surround mean of the features, scaled by its map
    maximum. Zero when the largest contrast is below 1e-9 of the largest feature norm (a flat map),
    so that rounding noise is not scaled up to 1."""
    features = vision_grid[0].detach().double().cpu()
    channels, height, width = features.shape
    flat = features.reshape(channels, height * width)
    centre = flat @ _gaussian_mean(height, width, CENTRE_SIGMA).T
    surround = flat @ _gaussian_mean(height, width, CENTRE_SIGMA * SURROUND_RATIO).T
    contrast = (centre - surround).norm(dim=0).reshape(height, width)
    top = float(contrast.max())
    if top <= 1e-9 * float(flat.norm(dim=0).max()):
        return torch.zeros(height, width)
    return (contrast / top).float()


def _root(parent: list, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def feature_segments(vision_grid: torch.Tensor, cosine: float = GROUP_COSINE):
    """[H, W] int labels, 0 for background. Two of the 8 neighbours link when the cosine of their
    features is at least `cosine`. A leader is linked to all its neighbours. A segment is a linked
    area with a leader (Wang and Terman 1997, as restated by Chen, Wang and Liu 2000)."""
    import numpy as np
    unit = F.normalize(vision_grid[0].detach().float().cpu(), dim=0)
    _, height, width = unit.shape
    parent = list(range(height * width))
    linked, neighbours = np.zeros(height * width, int), np.zeros(height * width, int)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for y in range(max(0, -dy), height - max(0, dy)):
            for x in range(max(0, -dx), width - max(0, dx)):
                a, b = y * width + x, (y + dy) * width + x + dx
                neighbours[[a, b]] += 1
                if float(unit[:, y, x] @ unit[:, y + dy, x + dx]) >= cosine:
                    linked[[a, b]] += 1
                    parent[_root(parent, a)] = _root(parent, b)
    roots = np.array([_root(parent, i) for i in range(height * width)])
    led = sorted({int(r) for r in roots[linked == neighbours]})
    labels = np.zeros(height * width, int)
    for number, root in enumerate(led, start=1):
        labels[roots == root] = number
    return labels.reshape(height, width)


class SegmentLimit:
    """Candidate K3: segments of the grey-level LEGION, then the segment with the most salience
    and sound (Quiles, Wang and colleagues 2011, abstract read) is selected."""

    def __init__(self, cosine: float = GROUP_COSINE):
        self.cosine = cosine

    def observe(self, vision_grid: torch.Tensor, audio_spatial: torch.Tensor) -> tuple:
        """(factors, record) from the encoder map and the sound direction."""
        height, width = vision_grid.shape[-2], vision_grid.shape[-1]
        labels = feature_segments(vision_grid, self.cosine)
        return select_group(labels, salience_drive(vision_grid), audio_drive(audio_spatial, height, width))
