# precondition-audit

A precondition audit for multi-environment causal representation learning:
does real perturbation data satisfy what the identifiability theorems assume?

This repository is self-contained. A fresh clone regenerates every table and
figure from committed artefacts, with no data and no external repository.

```bash
python scripts/88_make_descriptive_tables.py   # tables A and B
python scripts/90_spectrum_figure.py           # spectrum figures
python scripts/87_make_gate_tables.py          # gate tables
python scripts/89_make_index.py                # artefact index
```

| where | what |
|---|---|
| `scripts/` | diagnostics, profilers, table and figure generators |
| `results/descriptives/` | 8 dataset profiles + eigenvalue sidecars |
| `results/gates/` | simulator gate artefacts, by statistic |
| `results/preprocessing/` | empty; sweep artefacts pending, see its README |
| `paper/` | generated tables and figures |
| `manuscript/` | IEEE JBHI manuscript |
| `docs/` | reproduction, data sources, known limitations |

Read `docs/KNOWN_LIMITATIONS.md` before quoting any number. Raw data is not
redistributable and is never committed; see `docs/DATA_SOURCES.md`.

**No real-data rank-test result is claimed.** The rank test is gated, and the
gates showed it loses level control under nonlinear mixing, so it was never
run on real data. This repository reports descriptives and simulator gates only.
