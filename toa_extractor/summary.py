import os
import pandas as pd
import numpy as np
from pint.logging import log

from .utils.config import load_yaml_file, get_image_config
from .utils import process_and_copy_image


def process_images_for_summary(result_table, output_csv_path, config_file="default"):
    """
    Process and copy images for the summary, creating the images directory.

    Parameters
    ----------
    result_table : pandas.DataFrame
        The summary table containing image information
    output_csv_path : str
        Path to the output CSV file
    config_file : str
        Configuration file path
    """
    image_config = get_image_config(config_file)

    # Determine the base directory for images (relative to CSV output)
    csv_dir = os.path.dirname(os.path.abspath(output_csv_path))
    images_dir = os.path.join(csv_dir, image_config["directory"])

    # Track which images we've successfully processed
    processed_images = []

    for idx, row in result_table.iterrows():
        img_path = row.get("img_path", "")
        img_file = row.get("img_file", "")

        if img_path and img_file and os.path.exists(img_file):
            try:
                # Construct full target path
                target_path = os.path.join(csv_dir, img_path)

                # Process and copy the image
                process_and_copy_image(img_file, target_path, image_config)
                processed_images.append(img_path)
                log.info(f"Processed image: {img_file} -> {img_path}")

            except Exception as e:
                log.warning(f"Failed to process image {img_file}: {e}")
                # Set to empty string so plotting can handle missing image
                result_table.at[idx, "img_path"] = ""
        elif img_path:
            # Image path specified but source file doesn't exist
            log.warning(f"Source image file not found: {img_file}")
            result_table.at[idx, "img_path"] = ""

    # Remove the img_file column as it's no longer needed
    if "img_file" in result_table.columns:
        result_table.drop(columns=["img_file"], inplace=True)

    log.info(f"Successfully processed {len(processed_images)} images to {images_dir}")
    return result_table


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Create summary table for toaextract")

    parser.add_argument("files", help="Input binary files", type=str, nargs="+")
    parser.add_argument("-o", "--output", help="Output file name", type=str, default="summary.csv")
    parser.add_argument("-c", "--config", help="Configuration file", type=str, default="default")

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
        for arr in ["phase", "expo", "time"]:
            if arr in new_info:
                log.debug(f"Removing {arr} from metadata")
                del new_info[arr]

        newtab = pd.DataFrame(new_info)
        if len(newtab) == 0:
            continue
        if result_table is None:
            result_table = newtab
        else:
            result_table = pd.concat((result_table, newtab))

    if result_table is None or len(result_table) == 0:
        log.error("No valid data found in input files")
        return

    result_table.sort_values(by="mission", inplace=True)

    result_table["path"] = [os.path.dirname(f) for f in result_table["fname"]]
    result_table["fname"] = [os.path.basename(f) for f in result_table["fname"]]
    if "best_fit_amplitude_0" not in result_table or "best_fit_amplitude_1" not in result_table:
        log.warning("Missing amplitude columns.")
        ampl_to_noise = np.nan
    else:
        base = np.array(result_table["best_fit_amplitude_0"])
        peak = np.array(result_table["best_fit_amplitude_1"])
        scatter = np.sqrt(base + 0.75) + 1  # From Israel 1968, SRL internal report

        ampl_to_noise = peak / scatter
    result_table["amplitude_to_noise"] = ampl_to_noise

    # Process images if we have the new img_path column
    if "img_path" in result_table.columns:
        result_table = process_images_for_summary(result_table, args.output, args.config)

    result_table.to_csv(args.output)
