import os
import numpy as np
from astropy.table import Table, vstack


def get_toa_stats(summary_fname):
    """
    Get the TOA statistics from the summary file.
    """
    if not os.path.exists(summary_fname):
        raise FileNotFoundError(f"Summary file {summary_fname} does not exist.")
    table = Table.read(summary_fname)

    table_groups = table.group_by(["mission", "instrument"])
    for subtable in table_groups.groups:
        mission = subtable["mission"][0]
        instrument = subtable["instrument"][0]
        print(mission)
        # print(subtable["obsid", "mjd", "ephem", "fit_residual", "fit_residual_err"])
        subtable = subtable.group_by(["obsid", "mjd", "ephem"]).groups.aggregate(np.mean)
        # print(subtable["obsid", "mjd", "ephem", "fit_residual", "fit_residual_err"])
        subsubtable_list = []
        ephem_diff = []
        for subsub in subtable.group_by(["obsid", "mjd"]).groups:
            if len(subsub) > 1:
                ephem_diff.append(np.std(subsub["fit_residual"]))
            else:
                ephem_diff.append(np.nan)
            subsubtable = subsub["obsid", "mjd", "fit_residual", "fit_residual_err"]
            subsubtable = subsubtable.group_by(["obsid", "mjd"]).groups.aggregate(np.mean)
            subsubtable["mission"] = [mission] * len(subsubtable)
            subsubtable["instrument"] = [instrument] * len(subsubtable)
            subsubtable_list.append(subsubtable)

        subsubtable_aggr = vstack(subsubtable_list)
        subsubtable_aggr["ephem_std"] = ephem_diff

        print(subsubtable_aggr["obsid", "mjd", "fit_residual", "fit_residual_err", "ephem_std"])
        print(f"*** Results for mission {subsubtable_aggr['mission'][0]} and instrument {subsubtable_aggr['instrument'][0]} ***")
        print(f"Number of observations: {len(subsubtable_aggr)}")
        print(f"Mean residual: {np.nanmean(subsubtable_aggr['fit_residual']):.6f}")
        print(f"Standard dev: {np.nanstd(subsubtable_aggr['fit_residual']):.6f}")
        print(f"Mean stat err: {np.nanmean(subsubtable_aggr['fit_residual_err']):.6f}")
        print(f"Inter-ephem std: {np.nanmean(subsubtable_aggr['ephem_std']):.6f}")





def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Get TOA statistics from a summary file.")
    parser.add_argument("summary_fname", type=str, help="Path to the summary file.")
    args = parser.parse_args()

    get_toa_stats(args.summary_fname)