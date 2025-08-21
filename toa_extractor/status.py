import argparse
import os
from toa_extractor.pipeline import TOAPipeline


def main(args=None):
    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("files", help="Input event files", type=str, nargs="+")
    parser.add_argument("--config", help="Config file", type=str, default=None)
    parser.add_argument("-v", "--version", help="Version", type=str, default="none")
    parser.add_argument(
        "-o",
        "--unprocessed-output",
        help="Output file for unprocessed events",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--processed-output", help="Output file for processed events", type=str, default=None
    )
    parser.add_argument("--product-output", help="Output file for results", type=str, default=None)

    args = parser.parse_args(args)

    unprocessed_file_list = args.unprocessed_output
    if unprocessed_file_list is None:
        unprocessed_file_list = f"unprocessed_files_{args.version}.txt"
    processed_file_list = args.processed_output
    if processed_file_list is None:
        processed_file_list = f"processed_files_{args.version}.txt"
    product_file_list = args.product_output
    if product_file_list is None:
        product_file_list = f"product_files_{args.version}.txt"

    unproc_fobj = open(unprocessed_file_list, "w")
    proc_fobj = open(processed_file_list, "w")
    res_files = []

    processed = 0
    unprocessed = 0
    for fname in args.files:
        if not os.path.exists(fname):
            print(f"File {fname} does not exist. Skipping.")
            continue
        res_file = TOAPipeline(fname, args.config, args.version, 10).output().path
        if os.path.exists(res_file):
            print(fname, "-->", res_file)
            res_files.append(res_file)
            processed += 1
            print(res_file, file=proc_fobj)
        else:
            print(fname, "-->", "still to process")
            unprocessed += 1
            print(fname, file=unproc_fobj)

    unproc_fobj.close()
    proc_fobj.close()

    print(f"Processed {processed}/{len(args.files)} files.")
    print(f"{unprocessed} files still to process.")
