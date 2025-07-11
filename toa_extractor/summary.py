import os
import pandas as pd

from .utils.config import load_yaml_file


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Create summary table for toaextract")

    parser.add_argument("files", help="Input binary files", type=str, nargs="+")
    parser.add_argument("-o", "--output", help="Output file name", type=str, default="summary.csv")

    args = parser.parse_args(args)

    result_table = None
    for fname in args.files:
        print(f"Processing {fname}")
        info = load_yaml_file(fname)
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
    result_table.to_csv(args.output)
