#!/usr/bin/env python3
"""
Provenance script -- pulls every raw series the project needs from WRDS and
writes them to data/ as CSV.

This script is NOT part of the graded deliverable. The course brief requires
"a Python program IN A JUPYTER NOTEBOOK accompanied by ALL data required to
run the program", so the notebook must run offline, from the CSVs this script
produces. Keeping the pull separate is what makes that possible: a grader with
no WRDS account and no network can still execute the notebook top to bottom.

Run it once (or after changing the window):

    WRDS_USERNAME=yourid .venv/bin/python pull_data.py
    WRDS_USERNAME=yourid .venv/bin/python pull_data.py --force   # ignore cache

WINDOW. OptionMetrics is the binding constraint: optionm.opprcd stops at
2025-08-29, so the panel stops there too. Ending later would leave the five
option-derived features empty across the whole test set, which is exactly
where the brief's "at least half your features are not based on the target's
OHLCV bar" requirement is graded.

    2021-06-01  raw history starts (warms 60-day betas before the panel opens)
    2022-01-03  panel opens -- the first date all 20 names trade with warm
                rolling windows (WRBY IPO'd 2021-09-29 and is the binding name)
    2024-09-03  test set opens (final 12 months, held out)
    2025-08-29  panel closes (OptionMetrics cutoff)

UNIVERSE. The brief's own 20 suggested names, hardcoded. Not screened, not
re-selected. Any data-driven selection over the sample would choose names
using information from the end of the sample and then backtest across it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA = "data"

# The brief's 20, exactly as listed. Sector labels are the brief's own; the
# analysis uses CRSP siccd -> Fama-French 12 instead, but these are kept for
# cross-checking the deck.
MID_CAP = ["FLEX", "TWLO", "FTI", "CW", "NVT", "STRL", "XPO", "EWBC", "RGA", "EVR"]
SMALL_CAP = ["BLKB", "CNXC", "CACC", "LAZ", "EBC", "WSBC", "WRBY", "LQDA", "RSI", "USLM"]
UNIVERSE = MID_CAP + SMALL_CAP

# Benchmarks for the decoupling analysis in the EDA section. CRSP carries
# these as ordinary securities, so they come from the same table as the panel.
BENCHMARKS = ["SPY", "QQQ", "XLI"]

HIST_START = "2021-06-01"
PANEL_START = "2022-01-03"
TEST_START = "2024-09-03"
PANEL_END = "2025-08-29"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def path(name: str) -> str:
    return os.path.join(DATA, f"{name}.csv")


# Every date-like column written by this script, so a cached CSV reloads with
# the same dtypes it was saved with. Without this a cached frame comes back
# with string dates and silently fails to merge against a freshly pulled one.
DATE_COLS = ["date", "sdate", "edate", "datadate", "pub_date", "linkdt", "linkenddt"]


def cached(name: str, force: bool) -> pd.DataFrame | None:
    p = path(name)
    if force or not os.path.exists(p):
        return None
    head = pd.read_csv(p, nrows=0)
    df = pd.read_csv(p, parse_dates=[c for c in DATE_COLS if c in head.columns])
    print(f"  [cache] {name}: {len(df):,} rows")
    return df


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    p = path(name)
    df.to_csv(p, index=False)
    print(f"  [save]  {name}: {len(df):,} rows, {os.path.getsize(p) / 1e6:.1f} MB -> {p}")
    return df


def sql_str_tuple(items) -> str:
    """SQL IN clause for strings, safe for the 1-element case."""
    items = list(items)
    return "('" + "','".join(str(i) for i in items) + "')"


def sql_num_list(items) -> str:
    return ",".join(str(int(i)) for i in items)


# --------------------------------------------------------------------------
# Step 1 -- ticker -> permno, and the CRSP daily file
# --------------------------------------------------------------------------

def step1_prices(db, force=False) -> pd.DataFrame:
    banner("STEP 1  CRSP daily prices, quotes, sector")
    got = cached("prices", force)
    if got is not None:
        return got

    tickers = UNIVERSE + BENCHMARKS
    df = db.raw_sql(f"""
        SELECT permno, dlycaldt AS date, ticker, siccd,
               dlyprc, dlyopen, dlyhigh, dlylow, dlyclose,
               dlyvol, dlyprcvol, dlyret, dlyretx,
               dlybid, dlyask, dlycap, shrout,
               dlycumfacpr, dlycumfacshr
        FROM crsp.dsf_v2
        WHERE ticker IN {sql_str_tuple(tickers)}
          AND dlycaldt BETWEEN '{HIST_START}' AND '{PANEL_END}'
    """, date_cols=["date"])

    # A ticker string can be reused by a different company over a long window.
    # Over THIS window each of the 20 resolves to exactly one permno, so assert
    # it rather than silently aggregating two companies into one series.
    per_ticker = df.groupby("ticker")["permno"].nunique()
    ambiguous = per_ticker[per_ticker > 1]
    if len(ambiguous):
        raise RuntimeError(f"ticker -> permno is not 1:1 over the window: {ambiguous.to_dict()}")

    missing = sorted(set(UNIVERSE) - set(df["ticker"]))
    if missing:
        raise RuntimeError(f"CRSP has no rows for: {missing}")

    # Split/dividend-adjusted OHLCV. Without this the TA-Lib indicators run on
    # raw prices while the target runs on adjusted returns, and the two
    # disagree across every split and every dividend.
    for raw, adj in [("dlyopen", "open_adj"), ("dlyhigh", "high_adj"),
                     ("dlylow", "low_adj"), ("dlyclose", "close_adj")]:
        df[adj] = df[raw] / df["dlycumfacpr"]
    df["vol_adj"] = df["dlyvol"] * df["dlycumfacshr"]

    # The brief names this one explicitly: "bid-ask difference ... CRSP Daily
    # Stock File; look for bid and look for ask and take the difference".
    # Expressed as a relative quoted spread so it is comparable across price
    # levels; sign-guarded because CRSP stores a few crossed quotes.
    mid = (df["dlyask"] + df["dlybid"]) / 2.0
    df["stock_spread"] = np.where(mid > 0, (df["dlyask"] - df["dlybid"]) / mid, np.nan)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    uni = df[df["ticker"].isin(UNIVERSE)]
    print(f"  universe: {uni['ticker'].nunique()} names, {len(uni):,} rows, "
          f"{uni['date'].min():%Y-%m-%d} -> {uni['date'].max():%Y-%m-%d}")
    print(f"  benchmarks: {sorted(set(df['ticker']) & set(BENCHMARKS))}")
    print(f"  null quotes: {df['dlybid'].isna().sum():,} bid, {df['dlyask'].isna().sum():,} ask")
    print(f"  crossed/zero mid quotes -> NaN spread: {df['stock_spread'].isna().sum():,}")
    return save(df, "prices")


# --------------------------------------------------------------------------
# Step 2 -- permno -> secid, the OptionMetrics link
# --------------------------------------------------------------------------

def step2_optlink(db, prices: pd.DataFrame, force=False) -> pd.DataFrame:
    banner("STEP 2  CRSP -> OptionMetrics link")
    got = cached("optlink", force)
    if got is not None:
        return got

    permnos = (prices[prices["ticker"].isin(UNIVERSE)]
               .drop_duplicates("ticker")[["ticker", "permno"]])
    link = db.raw_sql(f"""
        SELECT secid, permno, sdate, edate, score
        FROM wrdsapps_link_crsp_optionm.opcrsphist
        WHERE permno IN ({sql_num_list(permnos['permno'])})
          AND sdate <= '{PANEL_END}' AND edate >= '{HIST_START}'
    """, date_cols=["sdate", "edate"])

    # THE BUG THIS FIXES. A permno can carry several secid links with
    # IDENTICAL sdate/edate -- TWLO has two. Tie-breaking on date overlap is
    # then a coin flip, and one of TWLO's two links (score 5) has ZERO rows in
    # optionm.opprcd while the other (score 1) has 62,516 in 2024 alone. Two
    # runs of the same code picked differently and the option-chain row count
    # moved 301,440 -> 677,582. WRDS scores link quality with 1 = best, so
    # order by score first and the ambiguity disappears entirely.
    link = link.sort_values(["permno", "score"]).drop_duplicates("permno", keep="first")
    link = permnos.merge(link, on="permno", how="left")

    unlinked = link[link["secid"].isna()]["ticker"].tolist()
    if unlinked:
        print(f"  WARNING: no OptionMetrics link for {unlinked} "
              f"-> option features stay NaN for these names")
    linked = link.dropna(subset=["secid"])
    print(f"  linked {len(linked)}/{len(link)} names, all at score "
          f"{sorted(linked['score'].unique())}")
    return save(link, "optlink")


# --------------------------------------------------------------------------
# Step 3 -- option chain features (IV, skew, gamma exposure, option spread)
# --------------------------------------------------------------------------

def step3_option_chain(db, prices: pd.DataFrame, link: pd.DataFrame, force=False) -> pd.DataFrame:
    banner("STEP 3  OptionMetrics chains -> IV / skew / GEX / quoted spread")
    got = cached("option_features", force)
    if got is not None:
        return got

    linked = link.dropna(subset=["secid"]).copy()
    linked["secid"] = linked["secid"].astype(int)
    secids = sorted(linked["secid"].unique())
    sec2tic = dict(zip(linked["secid"], linked["ticker"]))

    # Spot price for the gamma-exposure notional, from the same adjusted
    # series the features use.
    spot = (prices[prices["ticker"].isin(UNIVERSE)][["ticker", "date", "close_adj"]]
            .rename(columns={"close_adj": "spot"}))

    # These four features are QUOTE-based, not trade-based: implied vol, the
    # 25-delta skew, gamma exposure and the quoted spread all exist for any
    # contract the market is making a market in, whether or not it printed a
    # trade that day. Filtering the chain to `volume > 0` therefore throws
    # away most of a thinly-optioned name's chain -- it put WSBC at 4% IV
    # coverage and USLM at 7%. Open interest is the right liquidity gate.
    #
    # The bands are also widened from the textbook 45-55 / 20-30 delta to
    # 40-60 / 15-35, because small caps list far fewer strikes and often have
    # nothing inside the narrow band on a given day.
    #
    # Aggregation happens in SQL. The unfiltered chains run to millions of
    # contract-days per year; only the 20-names-by-date summary is needed, so
    # there is no reason to move the rest over the wire.
    y0, y1 = pd.Timestamp(HIST_START).year, pd.Timestamp(PANEL_END).year
    parts = []
    for y in range(y0, y1 + 1):
        try:
            agg = db.raw_sql(f"""
                SELECT secid, date,
                       AVG(impl_volatility) FILTER (
                           WHERE ABS(delta) BETWEEN 0.40 AND 0.60) AS iv_30d,
                       AVG(impl_volatility) FILTER (
                           WHERE cp_flag='P' AND ABS(delta) BETWEEN 0.15 AND 0.35) AS iv_put25,
                       AVG(impl_volatility) FILTER (
                           WHERE cp_flag='C' AND ABS(delta) BETWEEN 0.15 AND 0.35) AS iv_call25,
                       SUM(gamma * open_interest
                           * CASE WHEN cp_flag='C' THEN 1 ELSE -1 END) AS gex_raw,
                       AVG((best_offer - best_bid) / ((best_offer + best_bid) / 2.0)) FILTER (
                           WHERE ABS(delta) BETWEEN 0.40 AND 0.60
                             AND best_bid > 0 AND best_offer > best_bid) AS opt_spread,
                       COUNT(*) AS n_contracts
                FROM optionm.opprcd{y}
                WHERE secid IN ({sql_num_list(secids)})
                  AND date BETWEEN '{HIST_START}' AND '{PANEL_END}'
                  AND open_interest > 0
                GROUP BY secid, date
            """, date_cols=["date"])
        except Exception as e:
            print(f"    opprcd{y}: SKIPPED ({str(e).splitlines()[0][:80]})")
            continue
        print(f"    opprcd{y}: {len(agg):,} secid-days, "
              f"{int(agg['n_contracts'].sum()):,} contracts aggregated")
        parts.append(agg)

    if not parts:
        raise RuntimeError("no option chain data pulled -- check WRDS access to optionm")

    out = pd.concat(parts, ignore_index=True)
    out["ticker"] = out["secid"].map(sec2tic)
    out = out.merge(spot, on=["ticker", "date"], how="left")

    # 25-delta skew: put wing IV minus call wing IV. Needs BOTH wings quoted
    # that day, so it stays the sparsest of the four by construction.
    out["iv_skew"] = out["iv_put25"] - out["iv_call25"]

    # Net gamma exposure in dollars per 1% move. Dealers are taken to be long
    # calls / short puts, hence the sign. spot^2 * 0.01 converts share gamma
    # into the dollar notional that has to be hedged.
    out["gex"] = out["gex_raw"] * out["spot"] ** 2 * 0.01

    out = (out[["ticker", "date", "iv_30d", "iv_skew", "gex", "opt_spread", "n_contracts"]]
           .sort_values(["ticker", "date"]).reset_index(drop=True))

    print(f"  {len(out):,} ticker-days")
    for c in ["iv_30d", "iv_skew", "gex", "opt_spread"]:
        print(f"    {c:12s} non-null {out[c].notna().mean():5.1%}")
    return save(out, "option_features")


# --------------------------------------------------------------------------
# Step 4 -- daily put/call volume and open interest
# --------------------------------------------------------------------------

def step4_putcall(db, link: pd.DataFrame, force=False) -> pd.DataFrame:
    banner("STEP 4  put/call volume and open interest")
    got = cached("putcall", force)
    if got is not None:
        return got

    linked = link.dropna(subset=["secid"]).copy()
    linked["secid"] = linked["secid"].astype(int)
    sec2tic = dict(zip(linked["secid"], linked["ticker"]))

    # opvold is the pre-aggregated daily volume/OI table -- far cheaper than
    # re-scanning the full chains for the same numbers.
    df = db.raw_sql(f"""
        SELECT secid, date,
               SUM(CASE WHEN cp_flag='P' THEN open_interest ELSE 0 END) AS put_oi,
               SUM(CASE WHEN cp_flag='C' THEN open_interest ELSE 0 END) AS call_oi,
               SUM(CASE WHEN cp_flag='P' THEN volume ELSE 0 END) AS put_vol,
               SUM(CASE WHEN cp_flag='C' THEN volume ELSE 0 END) AS call_vol
        FROM optionm_all.opvold
        WHERE secid IN ({sql_num_list(sorted(sec2tic))})
          AND date BETWEEN '{HIST_START}' AND '{PANEL_END}'
        GROUP BY secid, date
    """, date_cols=["date"])
    df["ticker"] = df["secid"].map(sec2tic)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} ticker-days, {df['ticker'].nunique()} names")
    return save(df, "putcall")


# --------------------------------------------------------------------------
# Step 5 -- short interest, stamped at PUBLICATION date
# --------------------------------------------------------------------------

def step5_shortint(db, prices: pd.DataFrame, force=False) -> pd.DataFrame:
    banner("STEP 5  short interest (publication-lagged)")
    got = cached("shortint", force)
    if got is not None:
        return got

    permnos = (prices[prices["ticker"].isin(UNIVERSE)]
               .drop_duplicates("ticker")[["ticker", "permno"]])
    link = db.raw_sql(f"""
        SELECT gvkey, liid AS iid, lpermno AS permno, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE lpermno IN ({sql_num_list(permnos['permno'])})
          AND linktype IN ('LU','LC') AND linkprim IN ('P','C')
    """)
    pairs = link[["gvkey", "iid"]].drop_duplicates()
    conds = " OR ".join(f"(gvkey='{r.gvkey}' AND iid='{r.iid}')" for r in pairs.itertuples())

    df = db.raw_sql(f"""
        SELECT gvkey, iid, datadate, shortint, shortintadj
        FROM comp.sec_shortint
        WHERE ({conds}) AND datadate BETWEEN '{HIST_START}' AND '{PANEL_END}'
    """, date_cols=["datadate"])
    df = df.merge(link[["gvkey", "iid", "permno"]].drop_duplicates(), on=["gvkey", "iid"], how="left")
    df = df.merge(permnos, on="permno", how="left")

    # THE LEAK GUARD. datadate is the FINRA SETTLEMENT date; the figure is not
    # public for roughly 8 business days after it. Forward-filling from
    # settlement would show the model a number that did not exist yet, which
    # is a look-ahead that shows up as skill in the backtest.
    df["pub_date"] = df["datadate"] + pd.tseries.offsets.BDay(8)
    df = df.sort_values(["ticker", "pub_date"]).reset_index(drop=True)

    print(f"  {len(df):,} observations, {df['ticker'].nunique()}/{len(UNIVERSE)} names")
    print(f"  median publication lag: {(df['pub_date'] - df['datadate']).median().days} days")
    covered = sorted(df["ticker"].dropna().unique())
    if len(covered) < len(UNIVERSE):
        print(f"  no short-interest coverage: {sorted(set(UNIVERSE) - set(covered))}")
    return save(df, "shortint")


# --------------------------------------------------------------------------
# Step 6 -- market factors and VIX
# --------------------------------------------------------------------------

def step6_market(db, force=False) -> pd.DataFrame:
    banner("STEP 6  Fama-French factors and VIX")
    got = cached("market", force)
    if got is not None:
        return got

    ff = db.raw_sql(f"""
        SELECT date, mktrf, smb, hml, rf, umd FROM ff.factors_daily
        WHERE date BETWEEN '{HIST_START}' AND '{PANEL_END}'
    """, date_cols=["date"])
    vix = db.raw_sql(f"""
        SELECT date, vix FROM cboe.cboe
        WHERE date BETWEEN '{HIST_START}' AND '{PANEL_END}'
    """, date_cols=["date"])
    df = ff.merge(vix, on="date", how="outer").sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} dates | null mktrf {df['mktrf'].isna().sum()} | "
          f"null vix {df['vix'].isna().sum()}")
    return save(df, "market")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="ignore cached CSVs and re-pull")
    args = ap.parse_args()

    user = os.environ.get("WRDS_USERNAME")
    if not user:
        print("WRDS_USERNAME is not set.\n"
              "  WRDS_USERNAME=yourid .venv/bin/python pull_data.py", file=sys.stderr)
        return 1

    import wrds
    db = wrds.Connection(wrds_username=user)
    try:
        prices = step1_prices(db, args.force)
        link = step2_optlink(db, prices, args.force)
        step3_option_chain(db, prices, link, args.force)
        step4_putcall(db, link, args.force)
        step5_shortint(db, prices, args.force)
        step6_market(db, args.force)
    finally:
        db.close()

    banner("DONE")
    print(f"  window   {PANEL_START} -> {PANEL_END} (test opens {TEST_START})")
    print(f"  universe {len(UNIVERSE)} names, hardcoded from the course brief")
    print(f"  bundled  {DATA}/ -- the notebook reads these and needs no network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
