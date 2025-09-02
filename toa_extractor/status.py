import argparse
import os
from toa_extractor.data_setup import GetInfo
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
        "--output_yaml",
        help="Output file for list of yaml result files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--processed-output",
        help="List of processed files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial-output",
        help="Output file for list of partial result files",
        type=str,
        default=None,
    )
    args = parser.parse_args(args)

    unprocessed_file_list = args.unprocessed_output
    if unprocessed_file_list is None:
        unprocessed_file_list = f"unprocessed_files_{args.version}.txt"
    output_yaml_list = args.output_yaml
    if output_yaml_list is None:
        output_yaml_list = f"output_yaml_files_{args.version}.txt"
    partial_file_list = args.partial_output
    if partial_file_list is None:
        partial_file_list = f"partial_files_{args.version}.txt"
    processed_file_list = args.processed_output
    if processed_file_list is None:
        processed_file_list = f"processed_files_{args.version}.txt"

    unproc_fobj = open(unprocessed_file_list, "w")
    proc_fobj = open(output_yaml_list, "w")
    partial_fobj = open(partial_file_list, "w")
    processed_fobj = open(processed_file_list, "w")

    processed = 0
    unprocessed = 0
    partial = 0
    for fname in args.files:
        if not os.path.exists(fname):
            print(f"File {fname} does not exist. Skipping.")
            continue
        res_file = TOAPipeline(fname, args.config, args.version, 10).output().path
        info_file = GetInfo(fname, args.config, args.version, 10).output().path
        if os.path.exists(res_file):
            print(fname, "-->", res_file)
            yaml_files = list(filter(None, open(res_file, "r").read().splitlines()))
            for yaml_file in yaml_files:
                print(yaml_file, file=proc_fobj)
            print(fname, file=processed_fobj)
            processed += 1
        elif os.path.exists(info_file):
            print(fname, "-->", info_file, "(partial)")
            print(fname, file=partial_fobj)
            print(fname, file=unproc_fobj)  # Also to unprocessed files list!
            unprocessed += 1
            partial += 1
        else:
            print(fname, "-->", "still to process")
            unprocessed += 1
            print(fname, file=unproc_fobj)

    unproc_fobj.close()
    proc_fobj.close()
    partial_fobj.close()
    processed_fobj.close()

    print(f"Processed {processed}/{len(args.files)} files.")
    print(f"{unprocessed} files still to process.")
    print(f"of which {partial} partially processed.")
