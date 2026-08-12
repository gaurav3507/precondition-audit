"""Preprocessing-sweep tables: participation ratio and numerical rank by arm.

Reads only CURRENT artefacts under results/preprocessing/. Runs nothing.
Emits markdown + CSV to paper/tables/.

Two tables, same shape: rows = dataset, columns = the THREE preprocessing
options, never four.
  table_c_preprocessing_pr        participation ratio, plus a spread column
                                  (max/min across the three options)
  table_d_preprocessing_numrank   numerical rank at tol 1e-10, with the
                                  rank bound alongside

THE FOURTH ARM IS NOT A COLUMN. The sweep ran four arms but only three are
preprocessing choices. Every input matrix probes as already log-transformed
upstream, so log1p_std would be a double log; the sweep refused that arm and
recorded status "null" with a reason and no numbers. Rendering it as a fourth
body column, even filled with "n/a", presents a refusal as a missing
measurement and invites exactly the misreading the sweep exists to prevent. It
is emitted instead as a separate, labelled diagnostic block below each table.

Table notes travel with the tables, in the .md and the .csv, not as comments
in this file: a number quoted out of the table must be quotable with the
caveat that governs it.

Usage:
    python scripts/92_preprocessing_table.py
"""
import argparse
import csv
import datetime
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is
# one level up, not two. (It was two when these lived in
# causalbench/scripts/ in the source repo.)
REPO = HERE.parent
DEFAULT_DIR = REPO / "results/preprocessing"
OUT_DIR = REPO / "paper/tables"
NA = "n/a"

# The order the sweep declares in artefact["arms"]. Validated, not assumed.
ARMS = ("raw", "standardise", "log1p_std", "rank_int")
# The three that are preprocessing options. These, and only these, are columns.
BODY_ARMS = ("raw", "standardise", "rank_int")
# The diagnostic arm. Separate block, never a peer column.
DIAG_ARMS = tuple(a for a in ARMS if a not in BODY_ARMS)
# Per-dataset metadata carried alongside the arms. Constant across arms by
# construction (same control matrix, same cap), which is asserted at load.
META_COLS = ("n_genes", "n_control_used")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def script_commit():
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h",
                        "--", str(Path(__file__).relative_to(REPO))],
                       capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return NA if v is None else str(v)


def meta_of(doc):
    """Per-dataset metadata, asserted constant across the arms that ran.

    n_genes / n_control_used / rank_bound describe the control matrix, not the
    arm. If an arm disagrees the artefact is internally inconsistent and no
    table built from it means anything, so this is fatal rather than a
    silently-picked first value.
    """
    fields = META_COLS + ("rank_bound", "n_control_available")
    seen, out = {}, {}
    for arm, r in (doc.get("arms_result") or {}).items():
        if r.get("status") != "ok":
            continue
        for f in fields:
            seen.setdefault(f, {}).setdefault(r.get(f), []).append(arm)
    for f in fields:
        vals = seen.get(f) or {}
        if len(vals) > 1:
            raise SystemExit(
                f"[fatal] {doc.get('dataset')}: {f} differs across arms "
                f"{ {k: v for k, v in vals.items()} }. The arms are supposed to "
                f"share one control matrix; they do not. Refusing to tabulate.")
        out[f] = next(iter(vals), None)
    return out


def load(dirpath):
    kept, skipped = [], []
    for p in sorted(Path(dirpath).glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError as e:
            skipped.append((p.name, f"unparseable: {e}"))
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            skipped.append((p.name,
                            f"status={(doc.get('meta') or {}).get('status')!r}"))
            continue
        arms = tuple(doc.get("arms") or ())
        if arms != ARMS:
            raise SystemExit(
                f"[fatal] {p.name}: artefact declares arms {arms}, this table "
                f"is built for {ARMS}. An arm list that has moved silently is "
                f"how a diagnostic arm becomes a reported column. Refusing.")
        meta_of(doc)                      # fatal on internal inconsistency
        kept.append((p, doc))
    return kept, skipped


def build(kept, field, extra_cols=()):
    """One row per dataset: metadata, then the three preprocessing options.

    The spread column is max/min across those three and no others, so it
    answers "how much does the estimate move if you change preprocessing",
    which is the question the sweep was run to answer.
    """
    rows = []
    for _, doc in kept:
        res = doc.get("arms_result") or {}
        m = meta_of(doc)
        row = {"dataset": doc.get("dataset", NA)}
        for c in META_COLS:
            row[c] = m.get(c)
        for c in extra_cols:
            row[c] = m.get(c) if c in m else None
        vals = []
        for arm in BODY_ARMS:
            r = res.get(arm) or {}
            if r.get("status") == "ok" and r.get(field) is not None:
                row[arm] = r[field]
                vals.append(float(r[field]))
            else:
                row[arm] = NA
        if len(vals) == len(BODY_ARMS) and min(vals) > 0:
            row["spread_max_over_min"] = round(max(vals) / min(vals), 3)
        else:
            row["spread_max_over_min"] = NA
        rows.append(row)
    rows.sort(key=lambda r: str(r["dataset"]))
    return rows


def diagnostic_rows(kept):
    """The arms that are not preprocessing options, one row per dataset.

    Reports what the sweep actually did with the arm. Where it refused, the
    value columns say so in words rather than carrying a number or a bare
    blank that would read as a failed measurement.
    """
    rows = []
    for _, doc in kept:
        res = doc.get("arms_result") or {}
        for arm in DIAG_ARMS:
            r = res.get(arm) or {}
            status = r.get("status", "absent")
            rows.append({
                "dataset": doc.get("dataset", NA),
                "arm": arm,
                "status": status,
                "value_reported": ("none - arm refused" if status != "ok"
                                   else fmt(r.get("participation_ratio"))),
                "input_kind": doc.get("input_kind", "unknown"),
                "reason": r.get("reason") or NA,
            })
    rows.sort(key=lambda r: (str(r["arm"]), str(r["dataset"])))
    return rows


def rank_deficient(kept):
    """Datasets whose sample covariance is rank-deficient at the swept cap."""
    out = []
    for _, doc in kept:
        m = meta_of(doc)
        p, n = m.get("n_genes"), m.get("n_control_used")
        if p and n and p > n:
            out.append((doc.get("dataset"), round(p / n, 4)))
    return sorted(out)


def table_notes(kept, which):
    """Notes that must travel with the numbers. Same text in .md and .csv."""
    kinds = sorted({d.get("input_kind", "unknown") for _, d in kept})
    diag = ", ".join(DIAG_ARMS)
    rd = rank_deficient(kept)
    rd_txt = (", ".join(f"{d} (p/n = {r})" for d, r in rd) if rd else "none")
    caps = sorted({d.get("cap") for _, d in kept})
    cap = caps[0] if len(caps) == 1 else caps

    notes = [
        ("(a) The 'raw' arm is the control matrix exactly as the loader returns "
         "it, which is NOT raw counts. Every one of these matrices is already "
         "normalised by an undocumented upstream pipeline "
         f"(runtime probe: input_kind = {'/'.join(kinds)}; see "
         "docs/DATA_SOURCES.md). 'raw' records that this sweep applied no "
         "further transform, not that the input is untransformed."),
        (f"(b) '{diag}' is NOT a preprocessing option for these data and is not "
         "a column. Because the input is already log-like, it would be a "
         "double log. The sweep refused the arm and recorded no value for it. "
         "It appears only in the diagnostic block below, which reports the "
         "refusal and its reason."),
    ]
    if which == "pr":
        notes.append(
            f"(c) Rank deficiency at the swept cap n_control_used = {cap}: "
            f"{rd_txt}. Participation ratio is dominated by the leading "
            "eigenvalues and is the estimator intended for comparison across "
            "datasets, but where p > n the trailing spectrum is sampling "
            "noise and the value is still a sample-size-dependent lower bound. "
            "Compare arms within a row; compare rows only at matched n.")
    else:
        notes.append(
            f"(c) Rank deficiency at the swept cap n_control_used = {cap}: "
            f"{rd_txt}. Where p > n the numerical rank is bounded by "
            "rank_bound = n_control_used - 1 and is a property of the sample "
            "size, not of the data or of the preprocessing. Those entries are "
            "not comparable across datasets.")
        sat = all(r[a] == r["rank_bound"]
                  for r in build(kept, "numerical_rank", ("rank_bound",))
                  for a in BODY_ARMS
                  if isinstance(r[a], int))
        if sat:
            notes.append(
                "(d) Every entry in this table equals its own rank_bound. At "
                "tol 1e-10 the numerical rank saturates the bound under all "
                "three arms for all datasets, so the invariance shown here is "
                "arithmetic, not evidence that dimension is preprocessing-"
                "robust. Table C carries that evidence; this table records "
                "that the rank statistic does not discriminate at this cap.")
    return notes


def header_lines(kept, skipped, dirpath):
    commits = {}
    for p, doc in kept:
        commits.setdefault((doc.get("meta") or {}).get("git_commit", NA),
                           []).append(p.name)
    L = [f"generated  : {utc_now()}",
         f"generator  : 92_preprocessing_table.py @ {script_commit()}",
         f"source dir : {rel(dirpath)}",
         f"artefacts  : {len(kept)} CURRENT, {len(skipped)} skipped"]
    if len(commits) > 1:
        L.append("!! WARNING: MIXED PROVENANCE -- sources do not share one "
                 "meta.git_commit; rows below come from different versions.")
    for c, names in sorted(commits.items()):
        for n in sorted(names):
            L.append(f"  source   : {n}   meta.git_commit={c}")
    for n, why in skipped:
        L.append(f"  skipped  : {n}   ({why})")
    return L


DIAG_TITLE = ("DIAGNOSTIC ARM - NOT A PREPROCESSING OPTION, NOT PART OF THE "
              "TABLE ABOVE")


def write_outputs(name, rows, header, notes, diag, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    dcols = list(diag[0].keys()) if diag else []

    with open(outdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        for line in header:
            w.writerow([f"# {line}"])
        if cols:
            w.writerow(cols)
            for r in rows:
                w.writerow([fmt(r[c]) for c in cols])
        if dcols:
            w.writerow([])
            w.writerow([f"# {DIAG_TITLE}"])
            w.writerow(dcols)
            for r in diag:
                w.writerow([fmt(r[c]) for c in dcols])
        w.writerow([])
        for n in notes:
            w.writerow([f"# {n}"])

    L = [f"# {name}", "", "```"] + header + ["```", ""]
    if cols:
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    else:
        L.append("_no CURRENT preprocessing-sweep artefacts found_")
    L.append("")
    for n in notes:
        L += [f"> {n}", ""]
    if dcols:
        L += [f"### {DIAG_TITLE}", ""]
        L.append("| " + " | ".join(dcols) + " |")
        L.append("|" + "|".join(["---"] * len(dcols)) + "|")
        for r in diag:
            L.append("| " + " | ".join(fmt(r[c]) for c in dcols) + " |")
        L.append("")
    (outdir / f"{name}.md").write_text("\n".join(L))
    print(f"  wrote {name}.md / .csv   {len(rows)} rows, "
          f"{len(diag)} diagnostic rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--outdir", default=str(OUT_DIR))
    a = ap.parse_args()
    kept, skipped = load(a.dir)
    head = header_lines(kept, skipped, a.dir)
    print("\n".join(head))
    if not kept:
        print("\nno CURRENT artefacts; tables not written")
        return
    diag = diagnostic_rows(kept)
    write_outputs("table_c_preprocessing_pr",
                  build(kept, "participation_ratio"), head,
                  table_notes(kept, "pr"), diag, a.outdir)
    write_outputs("table_d_preprocessing_numrank",
                  build(kept, "numerical_rank", ("rank_bound",)), head,
                  table_notes(kept, "numrank"), diag, a.outdir)


if __name__ == "__main__":
    main()
