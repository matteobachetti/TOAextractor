import os
import numpy as np
from scipy.stats import median_abs_deviation
from astropy.table import Table, vstack
from astropy.io import ascii
from itertools import combinations


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
                combs = list(combinations(set(subsub["ephem"]), 2))
                diffs = {}
                for comb in combs:
                    comb = sorted(comb)
                    val = f"{comb[0]} - {comb[1]}"
                    comb0 = subsub[subsub["ephem"] == comb[0]]
                    comb1 = subsub[subsub["ephem"] == comb[1]]
                    diffs[val] = comb0["fit_residual"] - comb1["fit_residual"]
                    ephem_cols.append(val)

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
        n_meas = len(subsubtable_aggr)
        if n_meas < 3:
            mean_residual = np.nanmean(subsubtable_aggr["fit_residual"])
            std_residual = np.nan
            mean_stat_err = np.nanmean(subsubtable_aggr["fit_residual_err"])
        elif n_meas > 20:
            mean_residual = np.median(subsubtable_aggr["fit_residual"])
            std_residual = median_abs_deviation(subsubtable_aggr["fit_residual"], scale="normal")
            mean_stat_err = np.median(subsubtable_aggr["fit_residual_err"])
        else:
            mean_residual = np.nanmean(subsubtable_aggr["fit_residual"])
            std_residual = np.nanstd(subsubtable_aggr["fit_residual"])
            mean_stat_err = np.nanmean(subsubtable_aggr["fit_residual_err"])

        lines.append(
            [
                mission.upper(),
                instrument.upper(),
                n_meas,
                float(f"{mean_residual * 1e6:.1e}"),
                float(f"{std_residual * 1e6:.1e}"),
                float(f"{mean_stat_err * 1e6:.1e}"),
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
    final_table[names[:-1]].write(out_tex_fname, overwrite=True)
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
