# Audit of the thermodynamic track (2026-10-09 to 2026-10-11)

Date: 2026-10-13. Scope: the eleven merged changes that added the offline thermodynamic
probes and their result documents, the package `models/thermodynamic/`, and the public
page on thermodynamic computing. Checked on main `82c6b57`. This audit changes no
thermodynamic code and no result document. The corrections it lists are for the owner of
that track.

## Verdict, failures first

**Four checks FAILED. All four are about how results are presented. No number was found
wrong.**

| # | Check | Result |
|---|---|---|
| F1 | A correction is stated where the reader meets the claim | **FAILED in 9 documents.** Each of the 9 result documents written before the closed-stimulus finding has its correction as the last section (line 90 to 209). The first paragraph of each still gives the result as it was first read |
| F2 | A document that reports a PASS states that the replication failed | **FAILED.** `thermodynamic_transduction_gate_v3_2026_10.md` opens with "PASSED on `workspace_broadcast` and `obs_map`". Its own text names the replication that would test it. That replication FAILED on both streams (`thermodynamic_transduction_gate_v3_replication_2026_10.md`), and the gate v3 document does not say so |
| F3 | The status paragraph agrees with the rest of the document | **FAILED.** `docs/thermodynamic_computing_blueprint.md` says in its Status paragraph that no result from the package is reported. The same document reports five dated results further down, and `docs/results/` holds eleven result documents |
| F4 | The README test count follows the suite | **FAILED.** The README said 1534. The suite gives 1722 passed after these merges. A separate change corrects the README |

One wording point, not a failed check. `thermodynamic_frontier_2026_10.md` says under
prediction F1 that "the energy saving is real (4 times)". The energy in that document
is a count model (multiply-adds times an assumed ratio, plus channel events), the
ratio is not measured, and the 4 times is against the 256-axis encoding. The table
above that sentence states all of this. The sentence alone does not.

## What passed

| Check | Result | Evidence |
|---|---|---|
| Each pre-statement is older than its result | PASSED for 9 of 9 probe runs after gate v1 | Each pre-statement commit is older than its result commit. For the survival probe the commit was rebased after the run, so I also checked the push record of the clone. It shows the first push at 08:20, the first recording file is from 08:30, and the rebased commit has the same content. Push times of the other 8 were not checked |
| The same for gate v1 | NOT SHOWN | The probe with gate v1 and its result document entered the repo in one commit (`9e2d9ee`). The commit record cannot show that the gate was written before the run. The result was UNTESTABLE on all four streams, so no claim rests on it |
| No probe code changed between a pre-statement and its result | PASSED for 8 of 9. One change, disclosed | Gate v2 changed its coupling rule from an inverse to a pseudo-inverse after a first run stopped on a singular matrix. The gate v2 document states this |
| No exception is turned into a value | PASSED | One `try` in the package and the probes. It re-raises an `ImportError` with a message |
| Random draws are seeded | PASSED | No unseeded call found in the package or the probes |
| Assumed constants are marked | PASSED | The power, rate and efficiency constants are command line options with the word ASSUMED in their help text, and the result documents call the energy ratios arithmetic on assumed constants |
| The reference energy ratio has a source | PASSED | About 2 fJ per cell update is the modelled value of the vendor paper (arXiv 2510.23972, section III, read for this audit). The multiply-add figure is quoted second hand, and the document says so |
| Tests use closed forms or exact references | PASSED on inspection | 292 tests in 17 files. The sampler is tested against exact Boltzmann statistics, the channel against the Bernoulli variance, and the decoders against the project's own decoders with a raise on any difference |
| Production code does not import the package | PASSED | `tests/test_substrate_isolation.py`, 8 passed |
| Failures are stated first | PASSED | Gate v1 UNTESTABLE, gate v2 FAILED, gate v3 replication FAILED, one survival prediction FAILED. Each document opens with it |
| The public page agrees with the documents | PASSED | The page gives no result number from this track. Its instrument count (21, none trusted) equals `docs/instrument_inventory.md`. Its figures for the vendor chip equal the vendor page, and it labels them as vendor figures |

## Tests on two Python versions

| Environment | Thermodynamic, substrate and Brian2 tests (292) | Whole public suite |
|---|---|---|
| Python 3.8.3, torch 2.4.1 (the result machine) | 286 passed, 6 skipped | 1722 passed, 11 skipped |
| Python 3.12.10, torch 2.14.1, thrml 0.1.4, jax 0.11.2, nir 1.0.8 | 289 passed, 3 skipped | 1726 passed, 7 skipped |

On the result machine the 2 THRML tests and the 3 NIR tests skip, because those
libraries need Python 3.10 or later and are installed on no interpreter there. In a new
Python 3.12 environment they ran and passed. So the THRML adapter and the NIR export are
now checked with current library versions. On the result machine as found, no test of
them can run.

## Reproduction of one result

Document: `thermodynamic_survival_2026_10.md`. I recomputed the summary from the stored
per-row result files with the `summarise` command of the probe.

| Statement in the document | Recomputed |
|---|---|
| With pixel noise 8, shape decode on the project's split is 0.131 to 0.321 over all streams, models and both decoders | 0.131 to 0.321 |
| With whole stimuli held out, shape decode is 0.054 to 0.239 | 0.054 to 0.239 |
| No interval for that reading lies above chance | 0 of 18 intervals lie above chance |
| `obs_map` shape, linear decoder, no noise, raw input | 0.988 at models 42, 43 and 44 |
| `obs_map` shape reaches 90 percent of its clean value at 64 samples per axis at all three models | 64, 64, 64 |
| `obs_map` identity reaches it at 16 | 16, 16, 16 |

I also made the recording of model 42 again (no sensory noise, 280 trials, about 9
minutes) with the `record` command on the same revision of the code. All six arrays of
the new file are identical to the stored file, element by element (`obs_map`, `z_state`,
`tectum_content` and the three label arrays). So the stored recording comes from the
checkpoint and the code as stated.

## Not checked

- The 6-hour evaluation of the survival probe was not run again. The table above starts
  from its stored per-row results. Only the recording of model 42 without noise was made
  again.
- The numbers of the other ten result documents were not recomputed.
- No statement about hardware was tested. All of this track is software emulation.

## Corrections proposed to the owner of the track

1. Put one line at the top of each of the 9 earlier documents that names the
   closed-stimulus correction and links to it.
2. Put the replication result into the first lines of the gate v3 document.
3. Replace the Status paragraph of the blueprint with the present state.
4. In the frontier document, state in the F1 sentence that the energy is a count model
   and that the comparison is with the 256-axis encoding.
