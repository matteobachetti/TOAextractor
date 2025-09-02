import os
import re
import numpy as np
from scipy.stats import median_abs_deviation
from astropy.table import Table, vstack
from astropy.io import ascii
import astropy.units as u
from itertools import combinations
from .utils import KNOWN_OFFSETS


def write_pairwise_diffs(summary_fname, out_dir=None, max_dt_days=1.0):
    """
    Parse summary CSV and for each reference (mission,instrument) produce CSV files
    diff_REFM_REFI__OTHERM_OTHERI.csv listing all measurements from one specific
    other mission/instrument taken within max_dt_days of the reference mjd.

    Each output row contains:
      ref_mjd, other_mjd, diff_fit_residual (ref - other),
      diff_fit_residual_err (sqrt(err_ref^2 + err_other^2)),
      ref_mission, ref_instrument, other_mission, other_instrument

    Returns list of written filenames.
    """

    if out_dir is None:
        out_dir = os.getcwd()
    if not os.path.exists(summary_fname):
        raise FileNotFoundError(f"Summary file {summary_fname} does not exist.")

    table = Table.read(summary_fname)

    required = {"mjd", "mission", "instrument", "fit_residual", "fit_residual_err"}
    if not required.issubset(set(table.colnames)):
        missing = required - set(table.colnames)
        raise ValueError(f"Missing required columns in summary file: {missing}")

    mjds = np.array(table["mjd"], dtype=float)
    missions = np.array(table["mission"], dtype=str)
    instruments = np.array(table["instrument"], dtype=str)
    fit = np.array(table["fit_residual"], dtype=float)
    ferr = np.array(table["fit_residual_err"], dtype=float)

    n = len(table)
    # now key by ((ref_mission, ref_instrument), (other_mission, other_instrument))
    outputs = {}
    for i in range(n):
        if not np.isfinite(mjds[i]) or not np.isfinite(fit[i]) or not np.isfinite(ferr[i]):
            continue
        ref_mjd = mjds[i]
        # find other rows within time window (including same-day) but exclude same index
        close = np.abs(mjds - ref_mjd) <= float(max_dt_days)
        close[i] = False
        # exclude identical mission+instrument (we want different instrument/mission)
        same_pair = (missions == missions[i]) & (instruments == instruments[i])
        close = close & (~same_pair)
        idxs = np.where(close)[0]
        if idxs.size == 0:
            continue

        for j in idxs:
            if not np.isfinite(fit[j]) or not np.isfinite(ferr[j]) or not np.isfinite(mjds[j]):
                continue
            diff = fit[i] - fit[j]
            diff_err = np.sqrt(ferr[i] ** 2 + ferr[j] ** 2)
            key = ((missions[i], instruments[i]), (missions[j], instruments[j]))
            outputs.setdefault(key, [])
            outputs[key].append(
                (
                    float(ref_mjd),
                    float(mjds[j]),
                    float(diff),
                    float(diff_err),
                    missions[i],
                    instruments[i],
                    missions[j],
                    instruments[j],
                )
            )

    written_files = []
    for ((ref_mission, ref_instrument), (oth_mission, oth_instrument)), rows in outputs.items():
        if len(rows) == 0:
            continue
        names = [
            "ref_mjd",
            "other_mjd",
            "diff_fit_residual",
            "diff_fit_residual_err",
            "ref_mission",
            "ref_instrument",
            "other_mission",
            "other_instrument",
        ]
        out_table = Table(rows=rows, names=names)
        safe_ref_m = re.sub(r"\W+", "_", ref_mission.upper())
        safe_ref_i = re.sub(r"\W+", "_", ref_instrument.upper())
        safe_oth_m = re.sub(r"\W+", "_", oth_mission.upper())
        safe_oth_i = re.sub(r"\W+", "_", oth_instrument.upper())
        fname = os.path.join(
            out_dir, f"diff_{safe_ref_m}_{safe_ref_i}__{safe_oth_m}_{safe_oth_i}.csv"
        )
        out_table.write(fname, overwrite=True, format="csv")
        written_files.append(fname)

    return written_files


def get_toa_stats(
    summary_fname, out_fname="toa_stats_summary.csv", out_tex_fname="toa_stats_summary.tex"
):
    """
    Get the TOA statistics from the summary file.
    """
    if not os.path.exists(summary_fname):
        raise FileNotFoundError(f"Summary file {summary_fname} does not exist.")
    table = Table.read(summary_fname)

    table["rough_mjd"] = [float(f"{mjd:.2f}") for mjd in table["mjd"]]

    table_groups = table.group_by(["mission", "instrument"])
    lines = []

    for subtable in table_groups.groups:
        mission = subtable["mission"][0]
        instrument = subtable["instrument"][0]
        subtable = subtable.group_by(["obsid", "rough_mjd", "ephem"]).groups.aggregate(np.mean)
        subsubtable_list = []
        ephem_cols = []

        for subsub in subtable.group_by(["obsid", "rough_mjd"]).groups:
            print(subsub["rough_mjd", "ephem", "obsid"])
            if len(subsub) > 1:
                available_ephems = np.unique(subsub["ephem"])
                combs = list(combinations(available_ephems, 2))
                diffs = {}
                for eph in available_ephems:
                    subsub_eph = subsub[subsub["ephem"] == eph]
                    diffs[eph + "_fit_residual"] = subsub_eph["fit_residual"]
                    diffs[eph + "_fit_residual_err"] = subsub_eph["fit_residual_err"]
                    ephem_cols.extend([eph + "_fit_residual", eph + "_fit_residual_err"])

                for comb in combs:
                    comb = sorted(comb)
                    val = f"{comb[0]}_minus_{comb[1]}"
                    comb0 = subsub[subsub["ephem"] == comb[0]]
                    comb1 = subsub[subsub["ephem"] == comb[1]]
                    diffs[val] = comb0["fit_residual"] - comb1["fit_residual"]
                    diffs[val + "_err"] = np.sqrt(
                        comb0["fit_residual_err"] ** 2 + comb1["fit_residual_err"] ** 2
                    )

                    ephem_cols.extend([val, val + "_err"])

            else:
                diffs = {}
            subsubtable = subsub["obsid", "mjd", "rough_mjd", "fit_residual", "fit_residual_err"]
            subsubtable = subsubtable.group_by(["obsid", "rough_mjd"]).groups.aggregate(np.mean)
            for key, val in diffs.items():
                subsubtable[key] = val

            subsubtable["mission"] = [mission] * len(subsubtable)
            subsubtable["instrument"] = [instrument] * len(subsubtable)
            subsubtable_list.append(subsubtable)

        subsubtable_aggr = vstack(subsubtable_list)

        columns = ["obsid", "mjd", "fit_residual", "fit_residual_err"] + list(set(ephem_cols))
        print(subsubtable_aggr[columns])
        mission_table_fname = out_fname.replace(".csv", f"_{mission}_{instrument}.csv")
        subsubtable_aggr[columns].write(
            mission_table_fname,
            overwrite=True,
            format="csv",
            delimiter=",",
            fill_values=[(ascii.masked, "N/A")],
        )

        print(
            f"*** Results for mission {subsubtable_aggr['mission'][0]} and "
            f"instrument {subsubtable_aggr['instrument'][0]} ***"
        )
        print(f"Number of observations: {len(subsubtable_aggr)}")
        print(f"Mean residual (us): {1e6 * np.nanmean(subsubtable_aggr['fit_residual']):.2f}")
        print(f"Standard dev (us): {1e6 * np.nanstd(subsubtable_aggr['fit_residual']):.2f}")
        print(f"Mean stat err (us): {1e6 * np.nanmean(subsubtable_aggr['fit_residual_err']):.2f}")

        mission = subsubtable_aggr["mission"][0]
        instrument = subsubtable_aggr["instrument"][0]
        good = subsubtable_aggr["fit_residual_err"] < 0.015
        subsubtable_aggr = subsubtable_aggr[good]
        n_meas = len(subsubtable_aggr)
        offset = KNOWN_OFFSETS.get(
            f"{mission.lower()}/{instrument.lower()}",
            0 * u.us,
        ).to_value(u.s)

        print(f"{mission}/{instrument} offset (us): {1e6 * (offset):.2f}")
        label = "*" if np.abs(offset) > 0 else ""

        if n_meas < 3:
            mean_residual = np.nanmean(subsubtable_aggr["fit_residual"]) + offset
            std_residual = np.nan
            mean_stat_err = np.nanmean(subsubtable_aggr["fit_residual_err"])
        elif n_meas > 20:
            mean_residual = np.median(subsubtable_aggr["fit_residual"]) + offset

            std_residual = median_abs_deviation(subsubtable_aggr["fit_residual"], scale="normal")
            mean_stat_err = np.median(subsubtable_aggr["fit_residual_err"])
        else:
            mean_residual = np.nanmean(subsubtable_aggr["fit_residual"]) + offset

            std_residual = np.nanstd(subsubtable_aggr["fit_residual"])
            mean_stat_err = np.nanmean(subsubtable_aggr["fit_residual_err"])

        mean_residual_approx = float(f"{mean_residual * 1e6:.2e}")
        std_residual_approx = float(f"{std_residual * 1e6:.1e}")
        mean_stat_err_approx = float(f"{mean_stat_err * 1e6:.1e}")

        lines.append(
            [
                mission.upper(),
                instrument.upper(),
                n_meas,
                f"{mean_residual_approx:g}".replace("nan", "--") + label,
                f"{std_residual_approx:g}".replace("nan", "--"),
                f"{mean_stat_err_approx:g}".replace("nan", "--"),
            ]
        )

    names = [
        "Mission",
        "Instrument",
        "$N$",
        r"$r_{\rm mean}$ (us)",
        r"$\sigma$ (us)",
        r"$\sigma_{\rm stat}$ (us)",
    ]
    final_table = Table(
        rows=lines,
        names=names,
    )
    print("\nSummary of TOA statistics:")
    final_table.pprint()
    final_table[names].write(out_tex_fname, overwrite=True)
    final_table.write(out_fname, overwrite=True)
    return final_table


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Get TOA statistics from a summary file.")
    parser.add_argument("summary_fname", type=str, help="Path to the summary file.")
    parser.add_argument("-o", "--output", help="Output file name", type=str, default=None)
    args = parser.parse_args(args)

    if args.output is None:
        args.output = args.summary_fname.replace(".csv", "") + "_toa_stats.csv"

    get_toa_stats(
        args.summary_fname,
        out_fname=args.output,
        out_tex_fname=args.output.replace("csv", "tex"),
    )
    write_pairwise_diffs(
        args.summary_fname,
        out_dir=os.path.dirname(args.output),
    )
