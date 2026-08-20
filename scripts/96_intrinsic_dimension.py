"""Nonlinear intrinsic dimension of the control-cell matrices, matched to Table II.

WHY. The manuscript treats the observed control-covariance dimension as a lower
bound on latent dimension. That holds under linear noiseless mixing and fails
under nonlinear mixing: a one-dimensional latent embedded as (z, z^2) has a
rank-2 covariance and a manifold dimension of 1. So the covariance number can be
an order of magnitude too large without anything being wrong with it. This
measures the manifold dimension directly, at the same matched sample size, so
the paper can say whether the two agree in order of magnitude.

Precondition 3 is scoped and kept if intrinsic dimension lands near the
participation ratio, and reframed if it lands an order of magnitude lower.
NOTHING HERE IS TUNED TO FAVOUR EITHER OUTCOME, and the design that makes that
claim checkable is:

  * Every estimator that is run is reported, including ones that disagree.
    There is no preferred estimator and no post-hoc selection.
  * Neighbourhood parameters are fixed constants declared at the top of this
    file. MLE is reported across k in {5, 10, 20, 50} precisely because it is
    known to be k-sensitive, and reporting one k would hide that.
  * The correlation-dimension fit window is fixed by percentile of the pairwise
    distance distribution, stated up front, never chosen per dataset.
  * If the estimators disagree by more than a factor of two, that disagreement
    is the finding. It is recorded as `estimators_disagree` and stated in the
    summary rather than resolved by picking one.

STEP-0 SYNTHETIC GATE RUNS FIRST, ALWAYS. Three cases with known answers at the
shape of the real matrices. There is no flag that skips it: main() calls it
before the loaders are even imported, and a failure exits non-zero. Case 2 is
the whole point of the exercise. If the estimator returns the inflated
covariance rank on a nonlinear embedding of 10 latents, it cannot answer the
question this script exists to answer and the run stops there.

NO VERDICT ON THE DATA. This reports dimension estimates. It runs no rank test,
expresses no assumption verdict, and Phase B remains not authorised.

Usage (A100, cb venv). Long run; Claude Code cannot execute it:

    nohup python -u scripts/96_intrinsic_dimension.py \
        > logs/intrinsic_dimension.log 2>&1 &

    python scripts/96_intrinsic_dimension.py --selftest      # synthetic gate only
    python scripts/96_intrinsic_dimension.py --dataset k562

Env:
    PRECOND_DATA      raw data root; required for the real-data pass.
    PRECOND_EXTERNAL  third-party checkouts (Norman); see 94_control_pool_audit.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "results" / "intrinsic"
LOCKFILE = REPO / "logs" / "intrinsic_dimension.lock"

AUDIT = HERE / "94_control_pool_audit.py"          # control selection + provenance
SWEEP = HERE / "91_preprocessing_sweep.py"         # apply_arm, the two admissible arms
REQUIRED_SCRIPTS = (AUDIT, SWEEP,
                    HERE / "85_dataset_descriptives.py",
                    HERE / "81_ranktest_oracle.py")

# ---------------------------------------------------------------- parameters
# Fixed before any real data is touched. None of these is chosen per dataset.
N_GRID = (1000, 2000)              # Table II's matched size, and half of it
ARMS = ("raw", "standardise")      # the two admissible arms of Table III.
                                   # rank_int is deliberately NOT run: it is the
                                   # arm Table C note (d) shows concentrates the
                                   # spectrum into one direction on zero-inflated
                                   # input, and it is not a preprocessing choice
                                   # this paper endorses.
MLE_K = (5, 10, 20, 50)            # Levina-Bickel k, reported across all four
TWONN_DISCARD = 0.10               # Facco et al: drop the top 10% of mu ratios
CORRDIM_PCTL = (10.0, 30.0)        # correlation-dimension fit window, as
                                   # percentiles of the pairwise distance
                                   # distribution. Fixed, never per-dataset.
CORRDIM_NPTS = 20                  # log-spaced radii inside that window
BOOTSTRAP_N = 20                   # resamples for the spread; >= 20 as specified
BOOTSTRAP_FRAC = 0.80              # subsample fraction per resample
DISAGREE_FACTOR = 2.0              # max/min above this and the disagreement IS
                                   # the finding
SEED = 0

# ------------------------------------------------------- step-0 synthetic gate
# Known-answer cases at the shape of the real matrices. Expected values here are
# properties of constructions defined in this file, not of any real dataset.
SYN_N = 2000
SYN_P = 1158                       # k562's gene count, the narrowest real matrix
SYN_LATENT = 10
SYN_NOISE_SD = 0.01                # tiny, so the manifold is not blurred away
# Case 1 and 2 must land near SYN_LATENT. The band is wide because intrinsic
# dimension estimators are biased low on curved manifolds and this gate is
# checking fitness for purpose, not calibration.
SYN_REL_LO, SYN_REL_HI = 0.5, 2.5
# Case 2's failure mode is returning the COVARIANCE rank of the polynomial
# feature space instead of the manifold dimension. degree-2 features of d
# latents span d + d(d+1)/2 dimensions; anything at or above half of that is the
# estimator measuring the embedding, not the manifold.
SYN_POLY_DIM = SYN_LATENT + SYN_LATENT * (SYN_LATENT + 1) // 2      # 65
SYN_POLY_FAIL_AT = 0.5
# Case 3 has no ceiling to assert against: at n = 2000 in 1158 dimensions every
# neighbourhood estimator underestimates badly, by construction, because the
# sample can never fill the space. The honest assertion is that it separates
# from the structured cases by a wide margin, and that whatever it returns is
# RECORDED as this estimator's empirical ceiling at this (n, p).
SYN_CEILING_MULTIPLE = 3.0


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


# ================================================================= estimators
def _knn_dists(X, k_max):
    """Distances to the k_max nearest neighbours, self excluded. Exact, no tree.

    Chunked so a 2000 x 1158 float64 matrix never materialises an n x n float64
    block bigger than the chunk. Exactness matters more than speed here: an
    approximate neighbour search would be one more knob nobody could audit.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    sq = (X * X).sum(1)
    out = np.empty((n, k_max), dtype=np.float64)
    step = max(1, int(2e7 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        d2 = sq[a:b, None] - 2.0 * (X[a:b] @ X.T) + sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        for i in range(b - a):
            d2[i, a + i] = np.inf                      # exclude self
        part = np.partition(d2, k_max - 1, axis=1)[:, :k_max]
        part.sort(axis=1)
        out[a:b] = np.sqrt(part)
    return out


def twonn(X, discard=TWONN_DISCARD, dists=None):
    """TwoNN, Facco et al. 2017. ML on the ratio of the two nearest distances.

    mu_i = r2_i / r1_i is Pareto(1, d) under a locally uniform density, so
    -log(1 - F(mu)) is linear in log(mu) with slope d. Fitted through the
    origin on the lower (1 - discard) of the sorted ratios, which is the
    standard guard against the heavy right tail.
    """
    D = _knn_dists(X, 2) if dists is None else dists[:, :2]
    r1, r2 = D[:, 0], D[:, 1]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = mu.size
    keep = max(2, int(np.floor(n * (1.0 - discard))))
    mu = mu[:keep]
    mu = mu[mu > 1.0]
    if mu.size < 2:
        return None
    F = np.arange(1, mu.size + 1, dtype=np.float64) / (n + 1)
    x = np.log(mu)
    y = -np.log1p(-F)
    denom = float((x * x).sum())
    if denom <= 0:
        return None
    return float((x * y).sum() / denom)


def mle_levina_bickel(X, ks=MLE_K, dists=None):
    """Levina-Bickel MLE at each k, with the MacKay-Ghahramani correction.

    m_k(i)^-1 = mean_{j<k} log(T_k(i) / T_j(i)). Two aggregations are reported
    and neither is preferred: `avg` is the plain mean of the per-point m_k, and
    `inv_avg` inverts the mean of the inverses, which is the estimator MacKay
    and Ghahramani show is the consistent one. They differ, so both are shown.
    """
    D = _knn_dists(X, max(ks)) if dists is None else dists
    out = {}
    for k in ks:
        if k > D.shape[1] or k < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason=f"k={k} exceeds available neighbours")
            continue
        Tk = D[:, k - 1][:, None]
        Tj = D[:, :k - 1]
        ok = np.all(Tj > 0, axis=1) & (Tk[:, 0] > 0)
        if ok.sum() < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason="degenerate distances")
            continue
        inv_m = np.log(Tk[ok] / Tj[ok]).mean(axis=1)      # = 1 / m_k(i)
        good = inv_m > 0
        if good.sum() < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason="non-positive log ratios")
            continue
        m = 1.0 / inv_m[good]
        out[str(k)] = dict(avg=float(m.mean()),
                           inv_avg=float(1.0 / inv_m[good].mean()),
                           n_points_used=int(good.sum()))
    return out


def correlation_dimension(X, pctl=CORRDIM_PCTL, npts=CORRDIM_NPTS, rng=None):
    """Grassberger-Procaccia. Slope of log C(r) vs log r in a FIXED window.

    The window is the pctl[0]-pctl[1] percentile range of the pairwise distance
    distribution, identical for every dataset and arm. Choosing it per dataset
    is the classic way to make this estimator say whatever one wants, so it is
    not done. The fit's R^2 is reported so a bad linear region is visible.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    sq = (X * X).sum(1)
    lo_hi, counts, total = None, None, 0
    # one pass for the percentiles on a capped random pair sample, one for C(r)
    r = np.random.default_rng(SEED) if rng is None else rng
    m = min(n, 1500)
    idx = r.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Y = X[idx]
    sqy = (Y * Y).sum(1)
    d2 = sqy[:, None] - 2.0 * (Y @ Y.T) + sqy[None, :]
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(m, 1)
    dd = np.sqrt(d2[iu])
    dd = dd[dd > 0]
    if dd.size < npts:
        return dict(dimension=None, reason="too few positive distances")
    lo_hi = (float(np.percentile(dd, pctl[0])), float(np.percentile(dd, pctl[1])))
    if not (lo_hi[0] > 0 and lo_hi[1] > lo_hi[0]):
        return dict(dimension=None, reason="degenerate distance window")
    radii = np.logspace(np.log10(lo_hi[0]), np.log10(lo_hi[1]), npts)
    counts = np.zeros(npts, dtype=np.float64)
    step = max(1, int(2e7 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        blk = sq[a:b, None] - 2.0 * (X[a:b] @ X.T) + sq[None, :]
        np.maximum(blk, 0.0, out=blk)
        blk = np.sqrt(blk)
        for i in range(b - a):
            blk[i, a + i] = np.inf
        counts += (blk[:, :, None] < radii[None, None, :]).sum(axis=(0, 1))
        total += (b - a) * (n - 1)
    C = counts / max(total, 1)
    ok = C > 0
    if ok.sum() < 3:
        return dict(dimension=None, reason="C(r) zero across the window")
    x, y = np.log(radii[ok]), np.log(C[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(dimension=float(coef[0]),
                r2=(None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)),
                window=list(lo_hi), n_radii=int(ok.sum()),
                window_spec=(f"percentiles {pctl[0]}-{pctl[1]} of the pairwise "
                             f"distance distribution, fixed for every dataset"))


def all_estimators(X, rng=None):
    """Every estimator, always. Nothing is selected after seeing the answers."""
    D = _knn_dists(X, max(MLE_K))
    est = dict(
        twonn=twonn(X, dists=D),
        mle=mle_levina_bickel(X, dists=D),
        correlation_dimension=correlation_dimension(X, rng=rng),
    )
    est["point_estimates"] = point_estimates(est)
    vals = [v for v in est["point_estimates"].values() if v is not None and v > 0]
    if len(vals) >= 2:
        spread = max(vals) / min(vals)
        est["estimator_spread_max_over_min"] = float(spread)
        est["estimators_disagree"] = bool(spread > DISAGREE_FACTOR)
    else:
        est["estimator_spread_max_over_min"] = None
        est["estimators_disagree"] = None
    return est


def point_estimates(est):
    """Flatten to one number per estimator variant. Every variant appears."""
    out = {"twonn": est.get("twonn")}
    for k, rec in (est.get("mle") or {}).items():
        out[f"mle_k{k}_avg"] = rec.get("avg")
        out[f"mle_k{k}_inv_avg"] = rec.get("inv_avg")
    cd = est.get("correlation_dimension") or {}
    out["correlation_dimension"] = cd.get("dimension")
    return out


def bootstrap_spread(X, rng, n_boot=BOOTSTRAP_N, frac=BOOTSTRAP_FRAC):
    """Subsample spread per estimator variant. Reported as a range, not an SE.

    A percentile range says what the estimate did across resamples without
    asserting a sampling distribution these estimators do not have.
    """
    n = X.shape[0]
    m = max(20, int(round(frac * n)))
    acc = {}
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        pe = point_estimates(all_estimators(X[idx], rng=rng))
        for k, v in pe.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, vals in acc.items():
        good = [v for v in vals if v is not None]
        if not good:
            out[k] = dict(n=0, median=None, p05=None, p95=None, min=None, max=None)
            continue
        a = np.asarray(good, dtype=np.float64)
        out[k] = dict(n=int(a.size), median=float(np.median(a)),
                      p05=float(np.percentile(a, 5)),
                      p95=float(np.percentile(a, 95)),
                      min=float(a.min()), max=float(a.max()))
    return dict(n_resamples=int(n_boot), subsample_size=int(m),
                subsample_fraction=float(frac), per_estimator=out)


# ========================================================== step-0 synthetic
def _syn_cases(rng):
    """Three known-answer constructions at the shape of the real matrices."""
    Z = rng.standard_normal((SYN_N, SYN_LATENT))

    A1 = rng.standard_normal((SYN_LATENT, SYN_P))
    X1 = Z @ A1 + SYN_NOISE_SD * rng.standard_normal((SYN_N, SYN_P))

    # degree-2 polynomial features: the linear terms and every product z_i z_j
    cols = [Z]
    for i in range(SYN_LATENT):
        cols.append(Z[:, i:i + 1] * Z[:, i:])
    P = np.hstack(cols)
    P = (P - P.mean(0)) / np.where(P.std(0) > 0, P.std(0), 1.0)
    A2 = rng.standard_normal((P.shape[1], SYN_P))
    X2 = P @ A2 + SYN_NOISE_SD * rng.standard_normal((SYN_N, SYN_P))

    X3 = rng.standard_normal((SYN_N, SYN_P))

    return [
        dict(case="1_linear_gaussian", expected=float(SYN_LATENT), X=X1,
             description=(f"{SYN_LATENT} latents, random linear map into "
                          f"{SYN_P} columns")),
        dict(case="2_nonlinear_poly2", expected=float(SYN_LATENT), X=X2,
             description=(f"the same {SYN_LATENT} latents through a degree-2 "
                          f"polynomial feature map ({P.shape[1]} features) then "
                          f"a random linear embedding into {SYN_P} columns; the "
                          f"covariance rank is {P.shape[1]} and the manifold "
                          f"dimension is {SYN_LATENT}")),
        dict(case="3_isotropic_noise", expected=float(SYN_P), X=X3,
             description=(f"full-rank isotropic noise in {SYN_P} columns, no "
                          f"latent structure; no neighbourhood estimator can "
                          f"reach {SYN_P} at n={SYN_N}, so what is asserted is "
                          f"separation from cases 1 and 2 and what is recorded "
                          f"is this estimator's empirical ceiling")),
    ]


def step0_synthetic_gate():
    """ALWAYS RUNS. No flag skips it. Exits non-zero on any failure."""
    print("=" * 78, flush=True)
    print(" STEP-0 SYNTHETIC GATE  (known answers, no real data touched)",
          flush=True)
    print(f"   shape n={SYN_N} x p={SYN_P}, latent={SYN_LATENT}, "
          f"noise sd={SYN_NOISE_SD}", flush=True)
    print("=" * 78, flush=True)
    rng = np.random.default_rng(SEED)
    cases = _syn_cases(rng)
    results, fatal = [], []

    for c in cases:
        print(f"\n--- [{c['case']}] expected ~ {c['expected']:g} ---", flush=True)
        print(f"    {c['description']}", flush=True)
        est = all_estimators(c["X"], rng=rng)
        pe = est["point_estimates"]
        for name, v in pe.items():
            print(f"      {name:<24} {('None' if v is None else f'{v:.3f}')}",
                  flush=True)
        results.append(dict(case=c["case"], expected=c["expected"],
                            description=c["description"], estimators=est))

    by_case = {r["case"]: r for r in results}

    def headline(case):
        """TwoNN is the gate's headline because it has no free parameter."""
        return by_case[case]["estimators"]["point_estimates"].get("twonn")

    d1, d2, d3 = (headline("1_linear_gaussian"),
                  headline("2_nonlinear_poly2"),
                  headline("3_isotropic_noise"))

    for case, d in (("1_linear_gaussian", d1), ("2_nonlinear_poly2", d2)):
        if d is None:
            fatal.append(f"{case}: TwoNN returned no estimate.")
            continue
        lo, hi = SYN_REL_LO * SYN_LATENT, SYN_REL_HI * SYN_LATENT
        if not (lo <= d <= hi):
            fatal.append(
                f"{case}: TwoNN = {d:.3f}, outside the band "
                f"[{lo:g}, {hi:g}] around the known dimension {SYN_LATENT}.")

    if d2 is not None and d2 >= SYN_POLY_FAIL_AT * SYN_POLY_DIM:
        fatal.append(
            f"2_nonlinear_poly2: TwoNN = {d2:.3f}, at or above "
            f"{SYN_POLY_FAIL_AT:g} x the polynomial covariance rank "
            f"{SYN_POLY_DIM}. The estimator is measuring the embedding, not "
            f"the manifold. This is exactly the failure this script exists to "
            f"detect, so it cannot be used to answer the question. STOPPING.")

    if d3 is None:
        fatal.append("3_isotropic_noise: TwoNN returned no estimate.")
    elif d1 is not None and d3 < SYN_CEILING_MULTIPLE * d1:
        fatal.append(
            f"3_isotropic_noise: TwoNN = {d3:.3f}, less than "
            f"{SYN_CEILING_MULTIPLE:g} x the linear case ({d1:.3f}). The "
            f"estimator does not separate full-rank noise from a "
            f"{SYN_LATENT}-dimensional manifold at this (n, p).")

    print("\n" + "=" * 78)
    print(f" {'case':<24}{'expected':>10}{'TwoNN':>10}  verdict")
    print("-" * 78)
    for case, d, exp in (("1_linear_gaussian", d1, SYN_LATENT),
                         ("2_nonlinear_poly2", d2, SYN_LATENT),
                         ("3_isotropic_noise", d3, SYN_P)):
        bad = [f for f in fatal if f.startswith(case)]
        print(f" {case:<24}{exp:>10}"
              f"{('None' if d is None else f'{d:.2f}'):>10}  "
              f"{'FAIL' if bad else 'PASS'}")
    print("-" * 78)
    print(f" case 3 is recorded as the empirical ceiling at n={SYN_N}, "
          f"p={SYN_P}; it is not expected to reach {SYN_P}.")
    print("=" * 78)

    # Per-estimator calibration on the three known answers. This is what makes
    # the real-data table readable: an estimator that returns half the true
    # dimension on case 1 is expected to return half of it on real data too,
    # and a reader can see that here instead of guessing.
    calib = {}
    for r in results:
        for name, v in r["estimators"]["point_estimates"].items():
            calib.setdefault(name, {})[r["case"]] = v
    for name, byc in calib.items():
        lin = byc.get("1_linear_gaussian")
        byc["bias_vs_known_case1"] = (None if lin is None
                                      else float(lin) / SYN_LATENT)

    print(f"\n {'estimator':<24}{'case1 (10)':>12}{'case2 (10)':>12}"
          f"{'case3 (ceil)':>14}{'case1/known':>13}")
    print("-" * 78)
    for name in sorted(calib):
        b = calib[name]
        def f(x):
            return "None" if x is None else f"{x:.2f}"
        print(f" {name:<24}{f(b.get('1_linear_gaussian')):>12}"
              f"{f(b.get('2_nonlinear_poly2')):>12}"
              f"{f(b.get('3_isotropic_noise')):>14}"
              f"{f(b.get('bias_vs_known_case1')):>13}")
    print("-" * 78)

    gate = dict(shape=dict(n=SYN_N, p=SYN_P, latent=SYN_LATENT,
                           noise_sd=SYN_NOISE_SD),
                cases=results,
                calibration=calib,
                headline_estimator="twonn",
                band=dict(relative_lo=SYN_REL_LO, relative_hi=SYN_REL_HI),
                polynomial_covariance_rank=SYN_POLY_DIM,
                empirical_ceiling=d3,
                failures=fatal,
                passed=not fatal)
    if fatal:
        print("\n[fatal] STEP-0 SYNTHETIC GATE FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        print("        No real data was touched. Refusing to continue.",
              file=sys.stderr)
        sys.exit(1)
    print("\n  [gate] PASS\n", flush=True)
    return gate


# ================================================================= real data
def committed_pr(AUD, dataset, arm, n):
    """The committed participation ratio for this (dataset, arm, n), or None.

    raw          -> results/descriptives, latent_dimension_from_controls[n]
    standardise  -> results/preprocessing, arms_result.standardise
    Never a literal, and never a value from a different arm or a different n:
    an unmatched comparison is worse than no comparison.
    """
    if arm == "raw":
        val, src, cap = AUD.expected_pr(dataset, n)
        if val is None or cap != n:
            return None, None, "no committed raw PR at this n"
        return float(val), src, None
    for p in sorted((REPO / "results" / "preprocessing").glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        if doc.get("dataset") != dataset or int(doc.get("cap") or -1) != int(n):
            continue
        rec = (doc.get("arms_result") or {}).get("standardise") or {}
        if rec.get("status") == "ok" and rec.get("participation_ratio") is not None:
            return float(rec["participation_ratio"]), p.name, None
    return None, None, "no committed standardise PR at this n"


def run_condition(AUD, SW, DESC, dataset, args, gate):
    a = argparse.Namespace(hvg=args.hvg, seed=args.seed)
    Xc, rng, X_full, paths = AUD.control_matrix(DESC, dataset, a)
    n_all, p = Xc.shape
    state = rng.bit_generator.state          # same draw as Table II, per config
    prov = dict(
        resolved_input_paths=paths,
        input_path=(paths[0] if paths else None),
        n_cells=(int(X_full.shape[0]) if X_full is not None else None),
        n_genes=int(p),
        n_control_cells=int(n_all),
        control_value_array=AUD.value_fingerprint(Xc, "control matrix"),
    )
    out = []
    for n in N_GRID:
        if n > n_all:
            out.append(dict(dataset=dataset, n_requested=int(n),
                            status="skipped",
                            reason=f"only {n_all} control cells available"))
            continue
        rng.bit_generator.state = state
        idx = rng.choice(n_all, size=int(n), replace=False)
        base = np.asarray(Xc[idx], dtype=np.float64)
        for arm in ARMS:
            X = SW.apply_arm(base, arm)
            print(f"    [{dataset}] n={n} arm={arm} ...", flush=True)
            est = all_estimators(X, rng=rng)
            boot = bootstrap_spread(X, rng)
            pr, pr_src, pr_why = committed_pr(AUD, dataset, arm, n)
            pe = est["point_estimates"]
            ratios = {k: (None if pr in (None, 0) or v is None else float(v) / pr)
                      for k, v in pe.items()}
            out.append(dict(
                dataset=dataset, n_requested=int(n), n_used=int(n), arm=arm,
                status="ok",
                estimators=est,
                bootstrap=boot,
                participation_ratio=pr,
                participation_ratio_source=pr_src,
                participation_ratio_note=pr_why,
                intrinsic_over_participation=ratios,
                headline=dict(
                    twonn=pe.get("twonn"),
                    twonn_over_pr=ratios.get("twonn"),
                    estimator_spread=est.get("estimator_spread_max_over_min"),
                    estimators_disagree=est.get("estimators_disagree"),
                ),
            ))
            hp = out[-1]["headline"]
            print(f"        twonn={hp['twonn']}  PR={pr}  "
                  f"ratio={hp['twonn_over_pr']}  "
                  f"spread={hp['estimator_spread']}", flush=True)
    return prov, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="one condition only (default: all six)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic gate and stop")
    ap.add_argument("--hvg", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    # THE GATE RUNS FIRST, UNCONDITIONALLY. No flag reaches this line without
    # passing through it, and the loaders are not even imported until it has.
    gate = step0_synthetic_gate()
    if args.selftest:
        print("  --selftest: synthetic gate only, no real data touched.")
        return

    for q in REQUIRED_SCRIPTS:
        if not q.exists():
            sys.exit(f"[fatal] {q.name} is not in this repository (looked in "
                     f"{q.parent}). Not vendoring a copy and not "
                     f"reimplementing it.")
    AUD = _load(AUDIT, "_audit94")
    SW = _load(SWEEP, "_sweep91")
    todo = [args.dataset] if args.dataset else list(AUD.DATASETS)
    for d in todo:
        if d not in AUD.DATASETS:
            sys.exit(f"[fatal] unknown dataset {d!r}; expected one of "
                     f"{AUD.DATASETS}")
    AUD.step0_gate(todo)              # data presence, provenance, loud on miss

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
            sys.exit(f"[fatal] another run is live (pid {other}, lock "
                     f"{LOCKFILE}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        LOCKFILE.unlink()
    LOCKFILE.write_text(str(os.getpid()))

    try:
        out = Path(args.outdir)
        if not args.dataset and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        DESC = _load(HERE / "85_dataset_descriptives.py", "_descriptives")
        records, prov_by_ds, failed = [], {}, []
        for ds in todo:
            print(f"\n--- [{ds}] ---", flush=True)
            try:
                prov, recs = run_condition(AUD, SW, DESC, ds, args, gate)
            except Exception as e:                       # noqa: BLE001
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((ds, f"{type(e).__name__}: {e}"))
                continue
            prov_by_ds[ds] = prov
            records.extend(recs)

        want = len(todo) * len(N_GRID) * len(ARMS)
        got = sum(1 for r in records if r.get("status") == "ok")
        if failed or got != want:
            print(f"\n[fatal] SHORTFALL: {got} record(s) for {want} requested "
                  f"({len(todo)} conditions x {len(N_GRID)} sample sizes x "
                  f"{len(ARMS)} arms).", file=sys.stderr)
            for d, why in failed:
                print(f"  - {d}: {why}", file=sys.stderr)
            print("        This run is INCOMPLETE. Do not read the artefact as "
                  "a full sweep.", file=sys.stderr)
            sys.exit(1)

        disagree = [r for r in records
                    if r.get("status") == "ok" and r["headline"].get(
                        "estimators_disagree")]
        ts = DESC.utc_stamp()
        payload = dict(
            step0_synthetic_gate=gate,
            config=dict(n_grid=list(N_GRID), arms=list(ARMS),
                        mle_k=list(MLE_K), twonn_discard=TWONN_DISCARD,
                        corrdim_percentiles=list(CORRDIM_PCTL),
                        corrdim_n_radii=CORRDIM_NPTS,
                        bootstrap_n=BOOTSTRAP_N,
                        bootstrap_fraction=BOOTSTRAP_FRAC,
                        disagree_factor=DISAGREE_FACTOR, seed=args.seed,
                        rank_int_excluded=("deliberate: not an admissible "
                                           "preprocessing arm for these data")),
            provenance=prov_by_ds,
            records=records,
            n_configurations_with_estimator_disagreement=len(disagree),
            disagreement_note=(
                f"An estimator spread above {DISAGREE_FACTOR:g}x is reported as "
                f"the finding, not resolved by preferring one estimator. "
                f"{len(disagree)} of {got} configurations exceed it."),
            contains_no_test=True,
            disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no "
                        "assumption verdict is expressed or implied."),
            audit_script_sha256=AUD.sha256(AUDIT),
        )
        path = out / f"{ts.replace(':', '-')}__intrinsic_dimension.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[write] {path}", flush=True)

        print("\n" + "=" * 96)
        print(f" {'dataset':<20}{'n':>6}{'arm':<14}{'TwoNN':>9}"
              f"{'p05-p95':>18}{'PR':>11}{'ratio':>9}{'disagree':>10}")
        print("-" * 96)
        for r in records:
            if r.get("status") != "ok":
                print(f" {r['dataset']:<20}{r['n_requested']:>6}"
                      f"{'-':<14}{r['status']:>9}")
                continue
            b = (r["bootstrap"]["per_estimator"].get("twonn") or {})
            rng_s = ("-" if b.get("p05") is None
                     else f"{b['p05']:.2f}-{b['p95']:.2f}")
            pr = r["participation_ratio"]
            ra = r["headline"]["twonn_over_pr"]
            print(f" {r['dataset']:<20}{r['n_requested']:>6}{r['arm']:<14}"
                  f"{r['headline']['twonn']:>9.2f}{rng_s:>18}"
                  f"{('-' if pr is None else f'{pr:.1f}'):>11}"
                  f"{('-' if ra is None else f'{ra:.4f}'):>9}"
                  f"{str(r['headline']['estimators_disagree']):>10}")
        print("-" * 96)
        print(f" configurations with estimator spread > "
              f"{DISAGREE_FACTOR:g}x: {len(disagree)} of {got}")
        print("=" * 96)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


if __name__ == "__main__":
    main()
