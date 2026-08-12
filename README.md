# precondition-audit

Code, committed artefacts and manuscript for *A Precondition Audit of Benchmark
Datasets for Multi-Environment Causal Representation Learning*. Identifiability
theorems for multi-environment causal representation learning assume a supply
of distinct environments, a latent dimension small enough to estimate, and a
control covariance whose spectrum is not dominated by measurement noise. This
repository measures those three preconditions on six Perturb-seq control
matrices and two fMRI datasets, and reports where the assumptions are and are
not met. It is self-contained: a fresh clone rebuilds every table and figure in
the manuscript from committed artefacts, with no raw data and no external
repository. No real-data rank-test result is claimed anywhere in it.

## Quickstart

```bash
git clone https://github.com/gaurav3507/precondition-audit && cd precondition-audit
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` rebuilds `paper/tables/`, `paper/figures/` and `results/INDEX.md`
from committed artefacts and prints a PASS/FAIL table with before and after
artefact counts per stage. It exits non-zero if any stage produced nothing.
Add `--with-data` only on a machine that holds the raw matrices.

## What produces each manuscript table and figure

| manuscript item | produced by | reads |
|---|---|---|
| Table I, Environment Attrition (`tab:attrition`) | `scripts/88_make_descriptive_tables.py` | `results/descriptives/*.json` |
| Table II, Observed Dimension (`tab:dimension`) | `scripts/88_make_descriptive_tables.py` | `results/descriptives/*.json` |
| Table III, Preprocessing Sensitivity (`tab:preproc`) | `scripts/92_preprocessing_table.py` | `results/preprocessing/*.json` |
| Simulator gate tables (supplementary) | `scripts/87_make_gate_tables.py` | `results/gates/**/*.json` |
| Artefact index, `results/INDEX.md` | `scripts/89_make_index.py` | all of `results/` |
| Eigenvalue spectrum panels | `scripts/90_spectrum_figure.py` | `results/descriptives/*__spectra.npz` |
| Figure 1, environment breadth (`fig:breadth`) | no generator in this repository | `manuscript/goyal1.png` |
| Figure 2, observed dimension (`fig:dimension`) | no generator in this repository | `manuscript/goyal2.png` |
| Figure 3, preprocessing sensitivity (`fig:preproc`) | no generator in this repository | `manuscript/goyal3.png` |

The three manuscript figures are committed as rendered PNGs only. Their
generating scripts are not in this repository, so those three cannot be rebuilt
from a clone. The spectrum panels under `paper/figures/` are fully
reproducible but are not used in the manuscript.

## Hardware and runtime

Every committed table and figure was produced on an Apple Silicon Mac,
CPython 3.11.15, BLAS `accelerate`. Nothing in the Quickstart needs a GPU.

| stage | machine | wall clock | peak memory |
|---|---|---|---|
| `run_all.sh` (rebuild from artefacts) | any laptop | under 1 minute | under 1 GB |
| descriptives, 8 datasets (`--with-data`) | A100 node | about 1 hour 35 min | up to about 40 GB on Frangieh |
| preprocessing sweep, 6 datasets (`--with-data`) | A100 node | about 1 hour 10 min | up to about 40 GB on Frangieh |
| simulator gates 0, 1, 2 | any laptop | about 50, 20, 10 min | under 2 GB |

The Frangieh arms dominate both memory and runtime: 23712 genes by up to 6144
control cells, read as dense float32. The two data stages are launched
detached by `run_all.sh --with-data` for that reason.

## Data

No data is committed. None of it is redistributable.

| dataset | source | accession | DOI |
|---|---|---|---|
| Replogle K562, RPE1 | genome-scale Perturb-seq | CausalBench release npz | 10.1016/j.cell.2022.05.013 |
| Norman 2019 | CRISPRa genetic interaction screen | GEO GSE133344 | 10.1126/science.aax4438 |
| Frangieh, 3 arms | Perturb-CITE-seq | Single Cell Portal SCP1064 | 10.1038/s41588-021-00779-1 |
| HCP | 7 tasks, 2 encodings, 176 frames | Human Connectome Project | 10.1016/j.neuroimage.2013.05.041 |
| ABIDE | preprocessed timeseries | ABIDE Preprocessed Initiative | 10.1038/mp.2013.78 |

`docs/DATA_SOURCES.md` records what each loader does and states plainly that
the upstream normalisation of all six transcriptomic matrices is undocumented.

## The `$PRECOND_DATA` contract

No input path is hardcoded. Every loader resolves in this order and takes the
first entry that exists:

1. an explicit command-line argument
2. `$PRECOND_DATA`
3. `<repo>/data/`
4. `/workspace/precondition-audit/data`
5. `/workspace/ranktest-diagnostics/data`

A missing file is fatal and names every path tried, so a failed resolution is
never silent. ABIDE is the one exception: it reads `--abide-npz` then
`$ABIDE_NPZ` then its own candidate list, and does not consult
`$PRECOND_DATA`.

```bash
export PRECOND_DATA=/path/to/data
bash run_all.sh --with-data
```

Expected layout under `$PRECOND_DATA`: `dataset_k562.npz`, `dataset_rpe1.npz`,
`Norman2019_raw.h5ad`, the Frangieh SCP1064 CSVs, `hcp/ts/`, and
`abide_harmonized.npz`.

## What this repository does not do

It does not run a rank test on real data. The test is implemented and gated,
the gates showed it loses level control under nonlinear mixing, and it was
therefore never run on any real dataset. Phase B is deliberately absent.

It does not claim a causal conclusion. Every number here is descriptive: counts
of environments, dimension estimates from control covariance, spectrum shape,
and sensitivity to preprocessing. No assumption verdict is expressed or
implied, and the artefacts carry that disclaimer in their own payloads.

It does not establish that any dataset is adequate. Failing a precondition is
evidence against applicability; passing one is not evidence for it.

Read `docs/KNOWN_LIMITATIONS.md` before quoting any number.
