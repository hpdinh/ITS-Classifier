#!/usr/bin/env python3
import os
import json
import argparse
import yaml

from utils.data_utils import process_and_save

# --------------------
# Entrypoint
# --------------------
#../data/full_case_info.csv
#../data

def main():
    parser = argparse.ArgumentParser(description="Clean and split ServiceNow case data")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save outputs")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config with defunct and hyperparams"
    )
    args = parser.parse_args()
    # instead of json.load(f):
    with open(args.config, "r") as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            import json
            config = json.load(f)


    process_and_save(args, config)


if __name__ == "__main__":
    main()
