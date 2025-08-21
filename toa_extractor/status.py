import argparse
import os
from toa_extractor.pipeline import TOAPipeline


def main(args=None):
    parser = argparse.ArgumentParser(description="Calculate TOAs from event files")

    parser.add_argument("files", help="Input binary files", type=str, nargs="+")
    parser.add_argument("--config", help="Config file", type=str, default=None)
    parser.add_argument("-v", "--version", help="Version", type=str, default="none")
    parser.add_argument("-o", "--output", help="Output file", type=str, default=None)

    args = parser.parse_args(args)

    unprocessed_file_list = args.output
    if unprocessed_file_list is None:
        unprocessed_file_list = "unprocessed_files.txt"

    fobj = open(unprocessed_file_list, "w")
    res_files = []

    processed = 0
    unprocessed = 0
    for fname in args.files:
        res_file = TOAPipeline(fname, args.config, args.version, 10).output().path
        if os.path.exists(res_file):
            print(fname, "-->", res_file)
            res_files.append(res_file)
            processed += 1
        else:
            print(fname, "-->", "still to process")
            unprocessed += 1
            print(fname, file=fobj)

    fobj.close()
    print(f"Processed {processed}/{len(args.files)} files.")
    print(f"{unprocessed} files still to process.")
