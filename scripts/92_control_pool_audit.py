"""Is the n = 8000 panel of Fig. 2(b) honestly labelled?

The panel is captioned as a matched sample at n = 8000 control cells. That is
only true for a condition that HAS 8000 controls. 85_dataset_descriptives.py
subsamples with `n_use = int(min(cap, n_all))`, so a condition with fewer
controls silently contributes its whole pool at a smaller n and the panel is
then a mix of matched and unmatched points. This script measures which is which
and says so per condition, in one artefact.

READ AND RECOMPUTE, NOT A NEW MEASUREMENT. The control matrix comes from 85's
own loader, captured rather than re-derived, and the estimators are 85's own
`spectrum_estimators`. Wherever n_effective is 8000 the numbers must reproduce
what is already committed under results/descriptives/. The K562 step-0 gate is
what proves they did; it exits non-zero rather than reporting a drifted number.

NO VERDICT. This reports availability and matching. It expresses no assumption
verdict and Phase B remains not authorised.

Usage (A100, cb venv). This is a long run and Claude Code cannot execute it:

    nohup python -u scripts/92_control_pool_audit.py \
        > logs/control_pool_audit.log 2>&1 &

    python scripts/92_control_pool_audit.py --dataset k562   # one condition

Env:
    PRECOND_DATA   raw data root; required. Every input is existence-checked
                   against a candidate list and a miss names every path tried.
    ABIDE_NPZ      unused here (no fMRI condition has a control pool).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is one level
# up, not two. (It was two when these lived in causalbench/scripts/.)
REPO = HERE.parent
DESCRIPTIVES_DIR = REPO / "results" / "descriptives"
OUT_DIR = REPO / "results" / "regen" / "control_pool"
LOCKFILE = REPO / "logs" / "control_pool_audit.lock"

CAP = 8000                      # the cap the Fig. 2(b) panel claims
DATASETS = ("k562", "rpe1", "norman",
            "frangieh_coculture", "frangieh_control", "frangieh_ifng")

# The published K562 participation ratio at cap 8000, read off
# results/descriptives/2026-08-11T06-12-06Z__k562.json and pinned here so a
# drift is a loud failure rather than a new number nobody notices.
GATE_DATASET = "k562"
GATE_EXPECTED_PR = 552.5243597491797
GATE_TOL = 1e-10

# Files whose absence makes this script meaningless. The simulator is asserted
# even though this script does not import it: scripts A and B are a pair, and a
# repo missing the simulator cannot produce the paper's numbers at all. If it
# is absent this stops. It is never vendored from another repository and never
# reimplemented.
SIMULATOR = HERE / "81_ranktest_oracle.py"
REQUIRED_SCRIPTS = (SIMULATOR,
                    HERE / "85_dataset_descriptives.py",
                    HERE / "80_ranktest_core.py",
                    HERE / "84_results_io.py")

# Inputs, per condition, relative to a data root. Checked for existence before
# any work; a miss prints every root tried.
INPUTS = {
    "k562": ("dataset_k562.npz",),
    "rpe1": ("dataset_rpe1.npz",),
    "norman": ("Norman2019_raw.h5ad",),
    "frangieh_coculture": (), "frangieh_control": (), "frangieh_ifng": (),
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None       # loaders mkdir at import time
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_roots():
    """Ordered candidate roots. $PRECOND_DATA first, then the usual places."""
    roots = []
    if os.environ.get("PRECOND_DATA"):
        roots.append(("$PRECOND_DATA", Path(os.environ["PRECOND_DATA"])))
    roots += [("repo data/", REPO / "data"),
              ("A100 project", Path("/workspace/precondition-audit/data")),
              ("A100 legacy", Path("/workspace/ranktest-diagnostics/data"))]
    return roots


# ===================================================================== step 0
def step0_gate(datasets):
    """Everything that must be true before any measurement. Fails loudly."""
    print("=" * 70, flush=True)
    print(" STEP-0 GATE", flush=True)
    print("=" * 70, flush=True)
    fatal = []

    # ---- 1. $PRECOND_DATA
    root = os.environ.get("PRECOND_DATA")
    if not root:
        fatal.append("$PRECOND_DATA is not set.")
    else:
        print(f"  [env]  PRECOND_DATA = {root}", flush=True)
        if not Path(root).is_dir():
            fatal.append(f"$PRECOND_DATA is set but is not a directory: {root}")

    # ---- 2. every required input exists somewhere on the candidate list
    roots = data_roots()
    for ds in datasets:
        for rel in INPUTS.get(ds, ()):
            tried = [r / rel for _, r in roots]
            hit = next((p for p in tried if p.exists()), None)
            if hit is None:
                fatal.append(
                    f"input for {ds} not found: {rel}\n"
                    + "".join(f"        tried {lbl:<14} {r / rel}\n"
                              for lbl, r in roots).rstrip())
            else:
                print(f"  [data] {ds:<20} {rel:<24} -> {hit}", flush=True)
    # Frangieh is a directory of Single Cell Portal CSVs whose names 85 owns.
    # Assert the root resolves rather than second-guessing 85's filenames.
    if any(d.startswith("frangieh") for d in datasets):
        hit = next((r for _, r in roots if r.is_dir()), None)
        if hit is None:
            fatal.append("no data root on the candidate list is a directory; "
                         "Frangieh cannot resolve.\n"
                         + "".join(f"        tried {lbl:<14} {r}\n"
                                   for lbl, r in roots).rstrip())
        else:
            print(f"  [data] frangieh_*          (SCP1064 CSVs)        -> {hit}",
                  flush=True)

    # ---- 3. the simulator and its companions are IN THIS REPO
    for p in REQUIRED_SCRIPTS:
        if not p.exists():
            fatal.append(
                f"{p.name} is not in this repository (looked in {p.parent}).\n"
                "        STOPPING. Do not vendor a copy from another repo and\n"
                "        do not reimplement it: every number downstream depends\n"
                "        on that file being present and unmodified.")
        else:
            print(f"  [code] {p.name:<28} {sha256(p)}", flush=True)
    print(f"  [code] simulator resolved path : {SIMULATOR.resolve()}", flush=True)

    if fatal:
        print("\n[fatal] step-0 gate FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(2)
    print("  [gate] PASS\n", flush=True)


# ============================================================== control pools
def control_matrix(DESC, dataset, args):
    """The EXACT control matrix 85_dataset_descriptives feeds its estimators.

    Same capture 91_preprocessing_sweep.py uses: 85's own loader runs with
    latent_dim_block swapped for a grab, so the matrix and the rng arrive in
    precisely the state the estimators would have seen. Re-deriving the
    selection here would drift from 85 the first time either changed.
    """
    grabbed = {}

    def capture(Xc, rng, spectra=None, key_prefix=""):
        grabbed["X"] = np.asarray(Xc)
        grabbed["rng"] = rng
        return {}

    real = DESC.latent_dim_block
    DESC.latent_dim_block = capture
    try:
        base = dataset.split("_")[0]
        ns = argparse.Namespace(
            dataset=base, hvg=args.hvg, seed=args.seed,
            arm=(dataset.split("_", 1)[1] if dataset.startswith("frangieh_")
                 else None),
            abide_npz=None, n_spec_cap=None, overlap_check=None, selftest=False)
        rng = np.random.default_rng(args.seed)
        DESC.LOADERS[base](rng, ns, {})
    finally:
        DESC.latent_dim_block = real
    if "X" not in grabbed:
        sys.exit(f"[fatal] could not capture the control matrix for {dataset}")
    return grabbed["X"], grabbed["rng"]


def published_pr(dataset, cap):
    """The committed participation ratio for (dataset, cap), or None."""
    for p in sorted(DESCRIPTIVES_DIR.glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        label = doc.get("dataset")
        arm = (doc.get("arm") or (doc.get("meta") or {}).get("config", {}).get("arm"))
        if arm and not str(label).endswith(str(arm)):
            label = f"{label}_{arm}"
        if label != dataset:
            continue
        for blk in doc.get("blocks") or []:
            ld = blk.get("latent_dimension_from_controls") or {}
            rec = ld.get(str(cap))
            if rec and rec.get("participation_ratio") is not None:
                return float(rec["participation_ratio"]), p.name
    return None, None


def audit_one(DESC, dataset, args):
    Xc, rng = control_matrix(DESC, dataset, args)
    n_all, p = Xc.shape
    est = DESC.spectrum_estimators(Xc, CAP, rng)
    if est.get("error"):
        return dict(dataset=dataset, status="error", reason=est["error"],
                    available_controls=int(n_all), n_genes=int(p),
                    cap_requested=CAP)
    n_eff = int(est["n_control_used"])
    pub, pub_src = published_pr(dataset, CAP)
    rec = dict(
        dataset=dataset,
        status="ok",
        available_controls=int(n_all),
        n_effective=n_eff,
        cap_requested=CAP,
        matched=bool(n_eff == CAP),
        shortfall=int(max(0, CAP - n_eff)),
        n_genes=int(p),
        participation_ratio=float(est["participation_ratio"]),
        effective_rank_exp_spectral_entropy=float(
            est["effective_rank_exp_spectral_entropy"]),
        n_components_for_variance_95pct=int(
            est["n_components_for_variance"]["95pct"]),
        top_eigenvalue_share=float(est["top_eigenvalue_share"]),
        rank_bound=int(est["rank_bound"]),
        input_fingerprint=DESC.data_fingerprint(Xc),
        published_participation_ratio=pub,
        published_source=pub_src,
        reproduces_published=(None if pub is None
                              else bool(abs(float(est["participation_ratio"]) - pub)
                                        <= GATE_TOL)),
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS,
                    help="run one condition only (default: all six)")
    ap.add_argument("--hvg", type=int, default=None,
                    help="Frangieh gene cap; default unset, all genes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()
    todo = [a.dataset] if a.dataset else list(DATASETS)

    step0_gate(todo)

    # ------------------------------------------------------------- lockfile
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists():
        other = LOCKFILE.read_text().strip()
        alive = False
        if other.isdigit():
            try:
                os.kill(int(other), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            sys.exit(f"[fatal] another audit is live (pid {other}, "
                     f"lock {LOCKFILE}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        LOCKFILE.unlink()
    LOCKFILE.write_text(str(os.getpid()))

    try:
        # A stale directory must never let an exists-check report old numbers
        # as fresh. Only a full run clears it, so a per-dataset run appends.
        out = Path(a.outdir)
        if not a.dataset and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        DESC = _load(HERE / "85_dataset_descriptives.py", "_descriptives")
        records, failed = [], []
        for ds in todo:
            print(f"\n--- [{ds}] ---", flush=True)
            try:
                rec = audit_one(DESC, ds, a)
            except Exception as e:                       # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((ds, f"{type(e).__name__}: {e}"))
                continue
            records.append(rec)
            if rec["status"] != "ok":
                print(f"    {rec['status']}: {rec.get('reason')}", flush=True)
                continue
            print(f"    available={rec['available_controls']}  "
                  f"n_effective={rec['n_effective']}  "
                  f"matched={rec['matched']}  "
                  f"PR={rec['participation_ratio']:.6f}", flush=True)

            # ---- step-0 reproduction gate, K562 only
            if ds == GATE_DATASET:
                got = rec["participation_ratio"]
                if abs(got - GATE_EXPECTED_PR) > GATE_TOL:
                    sys.exit(
                        f"\n[fatal] STEP-0 REPRODUCTION GATE FAILED for {ds}\n"
                        f"        expected participation ratio "
                        f"{GATE_EXPECTED_PR!r}\n"
                        f"        got                        {got!r}\n"
                        f"        |diff| = {abs(got - GATE_EXPECTED_PR):.3e} "
                        f"> tol {GATE_TOL:g}\n"
                        f"        This is a recompute of a committed number, "
                        f"not a new measurement.\n"
                        f"        A disagreement means the control selection or "
                        f"the estimators have\n"
                        f"        moved, so every other row here is measuring "
                        f"the pipeline. Refusing.")
                print(f"    [gate] reproduces published PR "
                      f"{GATE_EXPECTED_PR!r}  PASS", flush=True)

        # ---- artefact assertion: N produced for N requested
        if failed or len(records) != len(todo):
            names = {r["dataset"] for r in records}
            missing = [d for d in todo if d not in names]
            print("\n[fatal] SHORTFALL: "
                  f"{len(records)} record(s) for {len(todo)} requested.",
                  file=sys.stderr)
            for d in missing:
                why = dict(failed).get(d, "no record produced")
                print(f"  - {d}: {why}", file=sys.stderr)
            print("        This run is INCOMPLETE. Do not read the artefact as "
                  "a full audit.", file=sys.stderr)
            sys.exit(1)

        ts = DESC.utc_stamp()
        payload = dict(
            cap_requested=CAP,
            datasets=list(todo),
            n_matched=sum(1 for r in records if r.get("matched")),
            n_unmatched=sum(1 for r in records
                            if r.get("status") == "ok" and not r["matched"]),
            records=records,
            contains_no_test=True,
            disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no "
                        "assumption verdict is expressed or implied."),
            simulator_sha256=sha256(SIMULATOR),
            simulator_path=str(SIMULATOR.relative_to(REPO)),
            peak_rss_mb=DESC.peak_rss_mb(),
        )
        path = out / f"{ts.replace(':', '-')}__control_pool_audit.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[write] {path}", flush=True)

        print("\n" + "=" * 70)
        print(f" {'dataset':<22}{'available':>11}{'n_eff':>8}{'matched':>9}"
              f"{'PR':>14}")
        print("-" * 70)
        for r in records:
            if r["status"] != "ok":
                print(f" {r['dataset']:<22}{r['available_controls']:>11}"
                      f"{'-':>8}{'-':>9}{r['status']:>14}")
                continue
            print(f" {r['dataset']:<22}{r['available_controls']:>11}"
                  f"{r['n_effective']:>8}{str(r['matched']):>9}"
                  f"{r['participation_ratio']:>14.4f}")
        print("-" * 70)
        print(f" matched at n={CAP}: {payload['n_matched']} of {len(records)}")
        print("=" * 70)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


if __name__ == "__main__":
    main()
