"""
P0-a — repair the filing<->text join (audit 2026-07-02).

Two verified defects in the original join (`topics/build_topic_docs.py` lookup on
`(ticker, fiscal_year)` == pickle filename year):

1. **Filename year is the FILING year, not the fiscal year.** Proof: 0/334 Dec-FYE `*_2019.pickle`
   files mention COVID (so they were filed early 2019 = FY2018 10-Ks), and only ~40% of `*_2020`
   pickles do (the Jan-Mar 2020 filing split). The old join therefore paired every text with the
   NEXT filing's row: labels/forward-vol windows start ~1 year after the text was published.
   Conservative (no leakage) but systematically stale — the fresh-text signal was never measured.
2. **`fiscal_year` (= report_date calendar year) is not unique per firm.** 68 (ticker, fiscal_year)
   groups hold two filings (52/53-week fiscal calendars, e.g. AAP FYE 2011-01-01 and 2011-12-31 both
   keyed 2011), so every (ticker, fiscal_year) merge cross-joined 2 texts x 2 rows.

Fix, written to NEW files (originals untouched):
  - `fiscal_year` re-keyed to the Compustat convention (FYE month Jan-May -> year-1); unique per firm.
  - `pickle_file` column: one-to-one pickle<->row assignment under the filing-year convention;
    multi-candidate cases (two filings in one calendar year) disambiguated by searching the row's
    report_date spelled out in the raw text, falling back to filing-order alignment.
  - Orphaned pickles from ticker renames (ABC->COR etc.) are REPORTED, not recovered here: their
    filings_index rows have has_pickle=False and no volatility labels were ever computed for them
    (needs a label recompute -- separate step).

Output:
  datasets/feature_table_fixed.parquet   (corrected table + pickle_file column)
  datasets/fix_filing_join_report.md     (diff report: match stats, dup resolution, orphans)

Run:  python dataset_config/fix_filing_join.py
"""

import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
SP500_DIR = ROOT / "datasets" / "sp500_1A"
OUT_TABLE = ROOT / "datasets" / "feature_table_fixed.parquet"
OUT_REPORT = ROOT / "datasets" / "fix_filing_join_report.md"

FNAME_RE = re.compile(r"^(.+)_(\d{4})\.pickle$")


def compustat_fyear(report_date: pd.Series) -> pd.Series:
    """Fiscal-year label, Compustat convention: FYE month Jan-May -> calendar year - 1."""
    return np.where(report_date.dt.month >= 6, report_date.dt.year, report_date.dt.year - 1)


def spelled_date_variants(d: pd.Timestamp) -> list[str]:
    """'October 28, 2007' style variants of a date, as they appear in 10-K text."""
    return [d.strftime("%B %-d, %Y"), d.strftime("%B %d, %Y")]


def disambiguate_by_text(fpath: Path, cands: pd.DataFrame) -> int | None:
    """Pick among candidate rows by finding the row's report_date spelled out in the raw text."""
    try:
        text = pickle.load(open(fpath, "rb"))
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    hits = []
    for idx, row in cands.iterrows():
        if any(v in text for v in spelled_date_variants(row.report_date)):
            hits.append(idx)
    return hits[0] if len(hits) == 1 else None


def main():
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["report_date"] = pd.to_datetime(ft["report_date"])
    n0 = len(ft)

    # ---- fix 2: unique fiscal-year key ------------------------------------------------
    old_dups = ft.duplicated(subset=["ticker", "fiscal_year"], keep=False).sum()
    ft["fiscal_year_orig"] = ft["fiscal_year"]
    ft["fiscal_year"] = compustat_fyear(ft["report_date"])
    rekeyed = int((ft["fiscal_year"] != ft["fiscal_year_orig"]).sum())
    new_dups = ft.duplicated(subset=["ticker", "fiscal_year"], keep=False)
    if new_dups.any():
        # residual collisions (e.g. two 10-Ks for the same true FY: amendments / index quirks)
        # keep the later filing (the authoritative one), drop the earlier
        ft = ft.sort_values(["ticker", "fiscal_year", "filing_date"])
        dropped = ft[ft.duplicated(subset=["ticker", "fiscal_year"], keep="last")]
        ft = ft.drop_duplicates(subset=["ticker", "fiscal_year"], keep="last")
        residual_dropped = len(dropped)
    else:
        residual_dropped = 0
    assert not ft.duplicated(subset=["ticker", "fiscal_year"]).any()

    # ---- fix 1: pickle <-> row assignment by FILING year -------------------------------
    ft["filing_year"] = ft["filing_date"].dt.year
    ft["pickle_file"] = pd.NA

    files = sorted(SP500_DIR.glob("*.pickle"))
    parsed = []
    for f in files:
        m = FNAME_RE.match(f.name)
        if m:
            parsed.append((f, m.group(1), int(m.group(2))))

    ft_tickers = set(ft["ticker"].unique())
    stats = Counter()
    orphan_tickers, gap_files, ambiguous_resolved, ambiguous_failed = Counter(), [], [], []

    by_key = {k: g for k, g in ft.groupby(["ticker", "filing_year"])}
    used_rows = set()
    for fpath, tk, fy in parsed:
        if tk not in ft_tickers:
            stats["orphan_ticker"] += 1
            orphan_tickers[tk] += 1
            continue
        cands = by_key.get((tk, fy))
        if cands is None:
            stats["no_row_for_filing_year"] += 1
            gap_files.append(fpath.name)
            continue
        cands = cands[~cands.index.isin(used_rows)]
        if len(cands) == 0:
            stats["row_already_taken"] += 1
            gap_files.append(fpath.name)
            continue
        if len(cands) == 1:
            idx = cands.index[0]
        else:
            idx = disambiguate_by_text(fpath, cands)
            if idx is None:
                idx = cands.sort_values("filing_date").index[0]  # in-order fallback
                ambiguous_failed.append(fpath.name)
            else:
                ambiguous_resolved.append(fpath.name)
        ft.loc[idx, "pickle_file"] = fpath.name
        used_rows.add(idx)
        stats["matched"] += 1

    # ---- old-vs-new comparison ----------------------------------------------------------
    # how many rows changed which text they carry? (old join: filename year == fiscal_year_orig)
    old_map = {}
    for fpath, tk, fy in parsed:
        old_map[(tk, fy)] = fpath.name  # last-wins, mirroring the old dict lookup
    ft["pickle_file_old"] = [old_map.get((t, y)) for t, y in zip(ft["ticker"], ft["fiscal_year_orig"])]
    both = ft["pickle_file"].notna() & ft["pickle_file_old"].notna()
    changed = int((ft.loc[both, "pickle_file"] != ft.loc[both, "pickle_file_old"]).sum())

    labelled = ft["fwd_vol_30d"].notna() & ft["lagged_vol_30d"].notna()
    lab_with_text_new = int((labelled & ft["pickle_file"].notna()).sum())
    lab_with_text_old = int((labelled & ft["pickle_file_old"].notna()).sum())
    split = np.select(
        [ft["filing_date"] < "2025-01-01", ft["filing_date"] < "2026-01-01"], ["train", "val"], "test")
    cov = pd.crosstab(split, labelled & ft["pickle_file"].notna())

    ft = ft.drop(columns=["pickle_file_old", "filing_year"])
    ft.to_parquet(OUT_TABLE, index=False)

    report = f"""# fix_filing_join report

## Fiscal-year re-key (Compustat convention: FYE Jan-May -> year-1)
- rows re-keyed: {rekeyed} of {n0}
- old duplicate (ticker, fiscal_year) rows: {old_dups} -> 0 after re-key
- residual same-FY duplicates dropped (kept later filing): {residual_dropped}

## Pickle <-> row assignment (filename year = FILING year; verified via COVID-mention test)
- pickle files: {len(parsed)}
- matched one-to-one: {stats['matched']}
- ambiguous (2 filings in one calendar year), resolved by report-date-in-text: {len(ambiguous_resolved)} {ambiguous_resolved}
- ambiguous, fell back to filing-order: {len(ambiguous_failed)} {ambiguous_failed}
- no table row for (ticker, filing year) [gap years, e.g. merger years / pre-2006 label universe]: {stats['no_row_for_filing_year']}
- orphaned by ticker rename (NOT recovered; needs label recompute): {stats['orphan_ticker']} files across {dict(orphan_tickers)}

## Impact vs old join
- rows whose text changed under the corrected join: {changed} (of {int(both.sum())} rows with text under both)
- labelled rows with text: old {lab_with_text_old} -> new {lab_with_text_new}
- labelled+text coverage by split (new):\n{cov.to_string()}

## Downstream invalidation
- `topics/data/topic_docs.jsonl` and every `topics/out/emb_*.npy` / topic-vector cache are aligned to the
  OLD join -> must be rebuilt (build_topic_docs against feature_table_fixed) and re-encoded (P0-b).
- All previous anchors (0.570 / 0.611 / 0.602) measured stale text; re-baseline after rebuild.
"""
    OUT_REPORT.write_text(report)
    print(report)
    print(f"Saved -> {OUT_TABLE}")
    print(f"Saved -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
