"""Is the n = 8000 panel of Fig. 2(b) honestly labelled, and where did each
number come from?

TWO QUESTIONS, ONE PASS.

(1) MATCHING. The panel is captioned as a matched sample at n = 8000 control
cells. That is only true for a condition that HAS 8000 controls.
85_dataset_descriptives.py subsamples with `n_use = int(min(cap, n_all))`, so a
condition with a smaller pool silently contributes all of it at a smaller n and
the panel is then a mix of matched and unmatched points. This reports
available_controls, n_effective and a matched boolean per condition.

(2) PROVENANCE. This is now the more valuable half. One published row currently
has no traceable source: three Norman files sit on the A100, all shaped
[108497, 5000], and which one produced the published participation ratio is not
recorded anywhere. Every record here therefore carries the resolved ABSOLUTE
input path and a content fingerprint of the value array, so no future row can
be un-sourceable in the same way.

NO PR LITERAL LIVES IN THIS FILE. The reproduction gate reads its expected
value out of the committed descriptives artefact for that dataset and cap. A
literal would be unsourceable in exactly the way this script exists to fix, and
would fail a correct computation the moment the artefact moved.

READ AND RECOMPUTE, NOT A NEW MEASUREMENT. The control matrix comes from 85's
own loader, captured rather than re-derived, and the estimators are 85's own
`spectrum_estimators`.

NO VERDICT. This reports availability, matching and provenance. It expresses no
assumption verdict and Phase B remains not authorised.

Usage (A100, cb venv). Long run; Claude Code cannot execute it:

    nohup python -u scripts/94_control_pool_audit.py \
        > logs/control_pool_audit.log 2>&1 &

    python scripts/94_control_pool_audit.py --dataset k562
    python scripts/94_control_pool_audit.py --norman-candidates

Env:
    PRECOND_DATA   raw data root; required. Every input is existence-checked
                   against a candidate list and a miss names every path tried.
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
FALLBACK_CAP = 2000             # every condition has an artefact at this cap
DATASETS = ("k562", "rpe1", "norman",
            "frangieh_coculture", "frangieh_control", "frangieh_ifng")

# The reproduction gate compares against the COMMITTED artefact, never a
# literal. Tolerance only; no participation-ratio value appears in this file.
GATE_DATASET = "k562"
GATE_TOL = 1e-10

# Norman: three files on the A100, all shaped [108497, 5000], and it is not
# recorded which produced the published number. All three are tried and all
# three are reported; this script does not pick one.
#
# Established before writing this list, and not re-derived here:
#   Norman2019_raw.h5ad and datasets/Norman2019.h5ad have byte-identical value
#   arrays (sha prefix 55ef3d07db20, max 8.1284); Norman2019_prep_new.h5ad
#   differs (5f33ad755e66, max 7.7348). None holds integral values, so the file
#   named "raw" is already log-transformed.
NORMAN_CANDIDATES = (
    "/workspace/external/discrepancy_vae/datasets/Norman2019_raw.h5ad",
    "/workspace/external/discrepancy_vae/datasets/datasets/Norman2019.h5ad",
    "/workspace/external/discrepancy_vae/datasets/datasets/Norman2019_prep_new.h5ad",
)
NORMAN_CANDIDATE_CAP = 2000     # the cap the published Norman number is quoted at

# Files whose absence makes this script meaningless. The simulator is asserted
# even though this script does not import it: scripts 93 and 94 are a pair, and
# a repo missing the simulator cannot produce the paper's numbers at all. If it
# is absent this stops. It is never vendored from elsewhere, never reimplemented.
SIMULATOR = HERE / "81_ranktest_oracle.py"
REQUIRED_SCRIPTS = (SIMULATOR,
                    HERE / "85_dataset_descriptives.py",
                    HERE / "80_ranktest_core.py",
                    HERE / "84_results_io.py")

INPUTS = {
    "k562": ("dataset_k562.npz",),
    "rpe1": ("dataset_rpe1.npz",),
    "norman": ("Norman2019_raw.h5ad",),
    "frangieh_coculture": (), "frangieh_control": (), "frangieh_ifng": (),
}

FP_NONZERO_N = 50_000           # entries hashed for the content fingerprint


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


def value_fingerprint(X, label):
    """Content fingerprint of a value array. Discriminates near-identical files.

    shape / dtype / max / mean, plus a sha256 over the first FP_NONZERO_N
    non-zero entries in row-major order. Non-zero because these matrices are
    mostly exact zeros: a hash over the leading raw entries would agree across
    files that differ everywhere it matters.
    """
    if X is None:
        return None
    A = np.asarray(X)
    flat = A.reshape(-1)
    nz = flat[flat != 0]
    take = nz[:FP_NONZERO_N]
    h = hashlib.sha256(np.ascontiguousarray(take, dtype=np.float64).tobytes())
    return dict(
        label=label,
        shape=[int(x) for x in A.shape],
        dtype=str(A.dtype),
        max=float(A.max()) if A.size else None,
        mean=float(A.mean()) if A.size else None,
        n_nonzero_hashed=int(take.size),
        nonzero_available=int(nz.size),
        sha256_first_nonzero=h.hexdigest(),
        hash_spec=(f"sha256 of the first {FP_NONZERO_N} non-zero entries, "
                   f"row-major, cast to float64"),
    )


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

    root = os.environ.get("PRECOND_DATA")
    if not root:
        fatal.append("$PRECOND_DATA is not set.")
    else:
        print(f"  [env]  PRECOND_DATA = {root}", flush=True)
        if not Path(root).is_dir():
            fatal.append(f"$PRECOND_DATA is set but is not a directory: {root}")

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
                print(f"  [data] {ds:<20} {rel:<24} -> {hit.resolve()}",
                      flush=True)
    if any(d.startswith("frangieh") for d in datasets):
        hit = next((r for _, r in roots if r.is_dir()), None)
        if hit is None:
            fatal.append("no data root on the candidate list is a directory; "
                         "Frangieh cannot resolve.\n"
                         + "".join(f"        tried {lbl:<14} {r}\n"
                                   for lbl, r in roots).rstrip())
        else:
            print(f"  [data] frangieh_*          (SCP1064 CSVs)        "
                  f"-> {hit.resolve()}", flush=True)

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


# ============================================================ expected values
def expected_pr(dataset, cap):
    """The committed participation ratio for (dataset, cap), read off disk.

    Returns (value, artefact_name, cap_used). Falls back to FALLBACK_CAP when
    the requested cap has no artefact, because the gate is a reproduction check
    and a check at a different cap still proves the pipeline has not moved.
    Never falls back to a literal: there is no literal to fall back to.
    """
    for want in (cap, FALLBACK_CAP):
        for p in sorted(DESCRIPTIVES_DIR.glob("*.json")):
            try:
                doc = json.load(open(p))
            except ValueError:
                continue
            if (doc.get("meta") or {}).get("status") != "CURRENT":
                continue
            label = doc.get("dataset")
            arm = doc.get("arm") or (doc.get("meta") or {}
                                     ).get("config", {}).get("arm")
            if arm and not str(label).endswith(str(arm)):
                label = f"{label}_{arm}"
            if label != dataset:
                continue
            for blk in doc.get("blocks") or []:
                ld = blk.get("latent_dimension_from_controls") or {}
                rec = ld.get(str(want))
                if rec and rec.get("participation_ratio") is not None:
                    return float(rec["participation_ratio"]), p.name, int(want)
    return None, None, None


# ============================================================== control pools
RESOLVER_NAMES = ("resolve_data", "_resolve", "norman_h5ad",
                  "frangieh_meta_csv", "frangieh_expr_csv")


def _wrap_resolvers(mod, seen, restore):
    """Wrap a loader's path resolvers so the resolved ABSOLUTE path is kept.

    Resolvers, not loaders: the resolver is the single point where a path
    becomes a file. Every wrap is recorded in `restore` and undone afterwards.
    """
    if mod is None:
        return
    for fname in RESOLVER_NAMES:
        real = getattr(mod, fname, None)
        if real is None or getattr(real, "_audit_wrapped", False):
            continue

        def wrapper(*a, _real=real, **kw):
            got = _real(*a, **kw)
            try:
                seen.append(str(Path(got).resolve()))
            except TypeError:
                pass
            return got

        wrapper._audit_wrapped = True
        setattr(mod, fname, wrapper)
        restore.append((mod, fname, real))


def control_matrix(DESC, dataset, args, norman_path=None):
    """The EXACT control matrix 85_dataset_descriptives feeds its estimators.

    Same capture 91_preprocessing_sweep.py uses: 85's own loader runs with
    latent_dim_block swapped for a grab, so the matrix and the rng arrive in
    precisely the state the estimators would have seen. Re-deriving the
    selection here would drift from 85 the first time either changed.

    Also grabs the FULL value array where 85 materialises one, and the absolute
    path of every file the loaders resolved. The resolvers are reached by
    intercepting 85's OWN `_load_module`: do_norman and do_frangieh each load a
    fresh module instance inside the call, so patching a separately-loaded copy
    would patch an object 85 never touches and would silently record nothing.
    03_screen is already bound as DESC.SCREEN at 85's import, so it is wrapped
    directly.

    Frangieh never materialises a full matrix by design: it reads only this
    arm's capped control cells, because RNA_expression.csv.gz is about 24.8 GB
    uncompressed. Its full-array fingerprint is null with a reason, not a
    fabricated number.
    """
    grabbed, paths, restore = {}, [], []

    def capture_ld(Xc, rng, spectra=None, key_prefix=""):
        grabbed["Xc"] = np.asarray(Xc)
        grabbed["rng"] = rng
        return {}

    real_ld = DESC.latent_dim_block
    real_pp = DESC.profile_perturb
    real_loadmod = DESC._load_module

    def capture_pp(X, iv, label, rng, spectra=None):
        grabbed["X_full"] = np.asarray(X)
        return real_pp(X, iv, label, rng, spectra)

    def loadmod(path, name):
        mod = real_loadmod(path, name)
        _wrap_resolvers(mod, paths, restore)
        if norman_path is not None and hasattr(mod, "norman_h5ad"):
            restore.append((mod, "norman_h5ad", mod.norman_h5ad))
            mod.norman_h5ad = lambda arg=None, _p=Path(norman_path): _p
            paths.append(str(Path(norman_path).resolve()))
        return mod

    base = dataset.split("_")[0]
    _wrap_resolvers(getattr(DESC, "SCREEN", None), paths, restore)
    DESC.latent_dim_block = capture_ld
    DESC.profile_perturb = capture_pp
    DESC._load_module = loadmod
    try:
        ns = argparse.Namespace(
            dataset=base, hvg=args.hvg, seed=args.seed,
            arm=(dataset.split("_", 1)[1] if dataset.startswith("frangieh_")
                 else None),
            abide_npz=None, n_spec_cap=None, overlap_check=None, selftest=False)
        rng = np.random.default_rng(args.seed)
        DESC.LOADERS[base](rng, ns, {})
    finally:
        DESC.latent_dim_block = real_ld
        DESC.profile_perturb = real_pp
        DESC._load_module = real_loadmod
        for mod, fname, real in reversed(restore):
            setattr(mod, fname, real)
    if "Xc" not in grabbed:
        sys.exit(f"[fatal] could not capture the control matrix for {dataset}")
    # dict.fromkeys keeps first-seen order and drops repeats
    return (grabbed["Xc"], grabbed["rng"], grabbed.get("X_full"),
            list(dict.fromkeys(paths)))


def measure(DESC, Xc, cap, rng):
    return DESC.spectrum_estimators(Xc, cap, rng)


def audit_one(DESC, dataset, args, cap=CAP, norman_path=None):
    Xc, rng, X_full, paths = control_matrix(DESC, dataset, args, norman_path)
    n_all, p = Xc.shape
    est = measure(DESC, Xc, cap, rng)

    full_fp = value_fingerprint(X_full, "full value array")
    if full_fp is None:
        full_fp = dict(
            label="full value array",
            unavailable=("85 never materialises a full matrix for this "
                         "condition: it reads only this arm's capped control "
                         "cells, because RNA_expression.csv.gz is about "
                         "24.8 GB uncompressed. Not fabricated."))
    prov = dict(
        resolved_input_paths=paths,
        input_path=(paths[0] if paths else None),
        n_cells=(int(X_full.shape[0]) if X_full is not None else None),
        n_genes=int(p),
        n_control_cells=int(n_all),
        full_value_array=full_fp,
        control_value_array=value_fingerprint(Xc, "control matrix"),
        legacy_fingerprint=DESC.data_fingerprint(Xc),
    )
    if est.get("error"):
        return dict(dataset=dataset, status="error", reason=est["error"],
                    available_controls=int(n_all), cap_requested=cap,
                    provenance=prov)

    n_eff = int(est["n_control_used"])
    exp, src, exp_cap = expected_pr(dataset, cap)
    got = float(est["participation_ratio"])
    return dict(
        dataset=dataset,
        status="ok",
        available_controls=int(n_all),
        n_effective=n_eff,
        cap_requested=cap,
        matched=bool(n_eff == cap),
        shortfall=int(max(0, cap - n_eff)),
        n_genes=int(p),
        participation_ratio=got,
        effective_rank_exp_spectral_entropy=float(
            est["effective_rank_exp_spectral_entropy"]),
        n_components_for_variance_95pct=int(
            est["n_components_for_variance"]["95pct"]),
        top_eigenvalue_share=float(est["top_eigenvalue_share"]),
        rank_bound=int(est["rank_bound"]),
        expected_participation_ratio=exp,
        expected_source=src,
        expected_at_cap=exp_cap,
        reproduces_expected=(None if exp is None or exp_cap != cap
                             else bool(abs(got - exp) <= GATE_TOL)),
        provenance=prov,
    )


# ========================================================= norman candidates
def norman_candidates(DESC, args):
    """All three Norman files at the published cap. Reports; never picks one."""
    cap = NORMAN_CANDIDATE_CAP
    exp, src, exp_cap = expected_pr("norman", cap)
    print("=" * 70, flush=True)
    print(f" NORMAN CANDIDATES at cap {cap}", flush=True)
    if exp is None:
        print("  no committed norman artefact at this cap; nothing to compare "
              "against", flush=True)
    else:
        print(f"  expected participation ratio {exp!r}", flush=True)
        print(f"  read from results/descriptives/{src} at cap {exp_cap}",
              flush=True)
    print("=" * 70, flush=True)

    out = []
    for cand in NORMAN_CANDIDATES:
        p = Path(cand)
        print(f"\n--- [{p.name}] ---\n    {p}", flush=True)
        if not p.exists():
            print("    ABSENT on this machine", flush=True)
            out.append(dict(path=str(p), status="absent"))
            continue
        try:
            rec = audit_one(DESC, "norman", args, cap=cap, norman_path=p)
        except Exception as e:                           # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            out.append(dict(path=str(p), status="error",
                            reason=f"{type(e).__name__}: {e}"))
            continue
        got = rec.get("participation_ratio")
        rec["candidate_path"] = str(p.resolve())
        rec["reproduces_published"] = (
            None if exp is None or got is None
            else bool(abs(got - exp) <= GATE_TOL))
        out.append(rec)
        fp = (rec["provenance"]["full_value_array"] or {})
        print(f"    PR={got!r}", flush=True)
        print(f"    reproduces published: {rec['reproduces_published']}",
              flush=True)
        print(f"    shape={fp.get('shape')} dtype={fp.get('dtype')} "
              f"max={fp.get('max')} mean={fp.get('mean')}", flush=True)
        print(f"    sha256(first {FP_NONZERO_N} nonzero)="
              f"{str(fp.get('sha256_first_nonzero'))[:12]}", flush=True)

    print("\n" + "=" * 70)
    print(f" {'file':<34}{'PR':>22}{'reproduces':>13}")
    print("-" * 70)
    for r in out:
        name = Path(r.get("candidate_path") or r["path"]).name
        if r.get("status") in ("absent", "error"):
            print(f" {name:<34}{r['status']:>22}{'-':>13}")
            continue
        print(f" {name:<34}{r['participation_ratio']:>22.10f}"
              f"{str(r['reproduces_published']):>13}")
    print("-" * 70)
    print(" All three are reported. This script does not pick one.")
    print("=" * 70)
    return dict(cap=cap, expected_participation_ratio=exp,
                expected_source=src, candidates=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS,
                    help="run one condition only (default: all six)")
    ap.add_argument("--norman-candidates", action="store_true",
                    help="try all three Norman files and report all three")
    ap.add_argument("--hvg", type=int, default=None,
                    help="Frangieh gene cap; default unset, all genes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()
    todo = [a.dataset] if a.dataset else list(DATASETS)
    if a.norman_candidates:
        todo = ["norman"]

    step0_gate(todo)

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
        out = Path(a.outdir)
        # A stale directory must never let an exists-check report old numbers
        # as fresh. Only a full run clears it, so a scoped run appends.
        if not a.dataset and not a.norman_candidates and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        DESC = _load(HERE / "85_dataset_descriptives.py", "_descriptives")

        if a.norman_candidates:
            block = norman_candidates(DESC, a)
            ts = DESC.utc_stamp()
            path = out / f"{ts.replace(':', '-')}__norman_candidates.json"
            path.write_text(json.dumps(
                dict(norman_candidates=block,
                     contains_no_test=True,
                     disclaimer=("DESCRIPTIVE ONLY. No rank test was run and "
                                 "no assumption verdict is expressed or "
                                 "implied."),
                     simulator_sha256=sha256(SIMULATOR),
                     peak_rss_mb=DESC.peak_rss_mb()), indent=2) + "\n")
            print(f"\n[write] {path}", flush=True)
            n_ok = sum(1 for c in block["candidates"]
                       if c.get("status") == "ok")
            if n_ok != len(NORMAN_CANDIDATES):
                print(f"\n[fatal] SHORTFALL: {n_ok} candidate(s) measured for "
                      f"{len(NORMAN_CANDIDATES)} requested. This run is "
                      f"INCOMPLETE.", file=sys.stderr)
                sys.exit(1)
            return

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
            print(f"    input     : {rec['provenance']['input_path']}",
                  flush=True)
            print(f"    available={rec['available_controls']}  "
                  f"n_effective={rec['n_effective']}  "
                  f"matched={rec['matched']}  "
                  f"PR={rec['participation_ratio']:.6f}", flush=True)

            if ds == GATE_DATASET:
                exp, src, exp_cap = (rec["expected_participation_ratio"],
                                     rec["expected_source"],
                                     rec["expected_at_cap"])
                if exp is None:
                    sys.exit(
                        f"\n[fatal] STEP-0 REPRODUCTION GATE CANNOT RUN for "
                        f"{ds}\n"
                        f"        No CURRENT descriptives artefact carries a "
                        f"participation ratio at\n"
                        f"        cap {CAP} or {FALLBACK_CAP}. There is no "
                        f"literal to fall back on, by design.\n"
                        f"        Looked in {DESCRIPTIVES_DIR}")
                if exp_cap == CAP:
                    got = rec["participation_ratio"]
                else:
                    # The artefact has no entry at CAP, so the gate falls back
                    # to the cap it does have. Recompute at that cap rather
                    # than compare across caps, which would always disagree.
                    Xc_g, rng_g, _, _ = control_matrix(DESC, ds, a)
                    got = float(measure(DESC, Xc_g, exp_cap,
                                        rng_g)["participation_ratio"])
                if abs(got - exp) > GATE_TOL:
                    sys.exit(
                        f"\n[fatal] STEP-0 REPRODUCTION GATE FAILED for {ds}\n"
                        f"        expected {exp!r}\n"
                        f"        got      {got!r}\n"
                        f"        |diff| = {abs(got - exp):.3e} > tol "
                        f"{GATE_TOL:g}\n"
                        f"        compared at cap {exp_cap}, read from "
                        f"results/descriptives/{src}\n"
                        f"        This is a recompute of a committed number, "
                        f"not a new measurement.\n"
                        f"        A disagreement means the control selection "
                        f"or the estimators have\n"
                        f"        moved, so every other row here is measuring "
                        f"the pipeline. Refusing.")
                print(f"    [gate] reproduces {src} at cap {exp_cap}: "
                      f"{exp!r}  PASS", flush=True)

        if failed or len(records) != len(todo):
            names = {r["dataset"] for r in records}
            missing = [d for d in todo if d not in names]
            print(f"\n[fatal] SHORTFALL: {len(records)} record(s) for "
                  f"{len(todo)} requested.", file=sys.stderr)
            for d in missing:
                print(f"  - {d}: {dict(failed).get(d, 'no record produced')}",
                      file=sys.stderr)
            print("        This run is INCOMPLETE. Do not read the artefact as "
                  "a full audit.", file=sys.stderr)
            sys.exit(1)

        ts = DESC.utc_stamp()
        payload = dict(
            cap_requested=CAP,
            fallback_cap=FALLBACK_CAP,
            datasets=list(todo),
            n_matched=sum(1 for r in records if r.get("matched")),
            n_unmatched=sum(1 for r in records
                            if r.get("status") == "ok" and not r["matched"]),
            records=records,
            norman_candidates_note=(
                "Three Norman files exist on the A100 and which produced the "
                "published number is not recorded. Run "
                "--norman-candidates to measure all three."),
            norman_candidate_paths=list(NORMAN_CANDIDATES),
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
