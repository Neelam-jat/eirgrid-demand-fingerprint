"""
The Data Centre Fingerprint in Ireland's Grid
---------------------------------------------
Analysis of 11 years (2014-2023) of EirGrid ROI system demand data
(15-minute SCADA readings, ~355k observations).

Data source: EirGrid Smart Grid Dashboard (smartgriddashboard.com),
downloaded via github.com/Daniel-Parke/EirGrid_Data_Download

Findings:
1. Average demand grew ~26% (2,942 MW -> 3,708 MW).
2. The night floor (3-5am) grew 39% while the evening peak (5-7pm)
   grew 21%. Always-on load is flattening the daily curve.
3. A forecasting model trained only on 2015-2019 under-forecasts
   later years by a growing, one-directional bias (+75 MW by 2023):
   the signature of a structural regime change.

Robustness:
- Weekend-only night floor shows the same 39% growth (rules out
  commercial daytime contamination).
- Floor share of peak rises +0.83 pp/year, r2=0.90, p=2.5e-05.
- SEAI independently reports data centres caused 88.2% of Irish
  demand growth since 2015 (metered data); this curve-shape method
  arrives at the same conclusion without seeing a single meter.

Author: Neelam Jat
"""

import glob

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_GLOB = "data/ROI_demandactual_*_Eirgrid.csv"


def load_demand(path_glob: str = DATA_GLOB) -> pd.DataFrame:
    """Load EirGrid 15-min system demand CSVs into an hourly series."""
    files = sorted(glob.glob(path_glob))
    df = pd.concat(
        pd.read_csv(f, header=None, names=["ts", "tag", "region", "mw"])
        for f in files
    )
    df["ts"] = pd.to_datetime(df["ts"], format="%d-%b-%Y %H:%M:%S")
    df = (
        df.dropna(subset=["mw"])
        .drop_duplicates("ts")
        .set_index("ts")
        .sort_index()
    )
    hourly = df["mw"].resample("h").mean().dropna().to_frame("demand")
    # dataset snapshot ends Feb 2024 -> keep full years only
    return hourly[hourly.index < "2024-01-01"]


def fingerprint(hourly: pd.DataFrame) -> pd.DataFrame:
    """Night floor vs evening peak, per year. The core finding."""
    night = hourly.between_time("03:00", "05:00")["demand"].resample("YE").mean()
    peak = hourly.between_time("17:00", "19:00")["demand"].resample("YE").mean()
    out = pd.DataFrame({"night_floor": night, "evening_peak": peak})
    out["floor_pct_of_peak"] = 100 * out["night_floor"] / out["evening_peak"]
    return out


def weekend_robustness(hourly: pd.DataFrame) -> pd.Series:
    """Same floor metric, weekends only (strips office/commercial load)."""
    weekend = hourly[hourly.index.dayofweek >= 5]
    return weekend.between_time("03:00", "05:00")["demand"].resample("YE").mean()


def trend_significance(fp: pd.DataFrame):
    """Linear trend on the floor's share of peak."""
    years = fp.index.year.values
    return stats.linregress(years, fp["floor_pct_of_peak"].values)


def old_world_model_test(hourly: pd.DataFrame) -> pd.DataFrame:
    """Train on 2015-2019, test on later years. Watch the bias grow."""
    df = hourly.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["doy"] = df.index.dayofyear
    df["lag24"] = df["demand"].shift(24)
    df["lag168"] = df["demand"].shift(168)
    df = df.dropna()

    feats = ["hour", "dow", "doy", "lag24", "lag168"]
    train = df[(df.index >= "2015-01-01") & (df.index < "2020-01-01")]
    model = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    model.fit(train[feats], train["demand"])

    rows = []
    for year in [2019, 2021, 2022, 2023]:
        t = df[df.index.year == year]
        err = t["demand"] - model.predict(t[feats])
        naive_err = t["demand"] - t["lag168"]
        rows.append(
            {
                "year": year,
                "ml_mae": err.abs().mean(),
                "ml_bias": err.mean(),
                "naive_mae": naive_err.abs().mean(),
                "naive_bias": naive_err.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("year")


if __name__ == "__main__":
    hourly = load_demand()
    print(f"{len(hourly):,} hourly observations, "
          f"{hourly.index.min():%Y-%m-%d} to {hourly.index.max():%Y-%m-%d}\n")

    fp = fingerprint(hourly)
    print("Night floor vs evening peak (MW):")
    print(fp.round(1).to_string(), "\n")

    g_floor = fp["night_floor"].iloc[-1] / fp["night_floor"].iloc[0] - 1
    g_peak = fp["evening_peak"].iloc[-1] / fp["evening_peak"].iloc[0] - 1
    print(f"2014->2023 growth: night floor +{g_floor:.0%}, "
          f"evening peak +{g_peak:.0%}\n")

    we = weekend_robustness(hourly)
    print(f"Weekend-only floor growth: +{we.iloc[-1]/we.iloc[0]-1:.0%}\n")

    reg = trend_significance(fp)
    print(f"Floor share trend: {reg.slope:+.2f} pp/year, "
          f"r2={reg.rvalue**2:.3f}, p={reg.pvalue:.2e}\n")

    print("Old-world model test (trained 2015-2019):")
    print(old_world_model_test(hourly).round(1).to_string())
