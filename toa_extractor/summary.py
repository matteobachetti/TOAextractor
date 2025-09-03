import os
import pandas as pd
import numpy as np
from pint.logging import log

from .utils.config import load_yaml_file


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Create summary table for toaextract")

    parser.add_argument("files", help="Input binary files", type=str, nargs="+")
    parser.add_argument("-o", "--output", help="Output file name", type=str, default="summary.csv")

    args = parser.parse_args(args)

    result_table = None
    for fname in args.files:
        log.info(f"Processing {fname}")
        if not os.path.exists(fname):
            log.warning(f"File {fname} does not exist.")
            continue
        info = load_yaml_file(fname)
        if info is None:
            log.warning(f"File {fname} could not be read.")
            continue
        new_info = dict([(key, [val]) for key, val in info.items()])
        for arr in ["phase", "expo"]:
            if arr in new_info:
                del new_info[arr]

        newtab = pd.DataFrame(new_info)
        if len(newtab) == 0:
            continue
        if result_table is None:
            result_table = newtab
        else:
            result_table = pd.concat((result_table, newtab))
    result_table.sort_values(by="mission", inplace=True)

    result_table["path"] = [os.path.dirname(f) for f in result_table["fname"]]
    result_table["fname"] = [os.path.basename(f) for f in result_table["fname"]]
    if "best_fit_amplitude_0" not in result_table or "best_fit_amplitude_1" not in result_table:
        log.warning("Missing amplitude columns.")
        ampl_to_noise = np.nan
    else:
        ampl_to_noise = result_table["best_fit_amplitude_1"] / max(
            1, np.sqrt(result_table["best_fit_amplitude_0"])
        )
    result_table["amplitude_to_noise"] = ampl_to_noise
    result_table.to_csv(args.output)
