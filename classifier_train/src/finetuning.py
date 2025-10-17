import argparse
import os
from datetime import datetime
import json
import yaml

from utils.train_utils import train, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ticket Classifier Training Script")

    # add config arg
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")

    # define hyperparameters (same as before, but optional)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--task", type=str, default="ticketclass")
    parser.add_argument("--project", type=str, default="ticket-classifier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=2250)
    parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    # Parse CLI args
    args = parser.parse_args()
    args_dict = vars(args)

    # If config file given, load and override defaults
    if args.config:
        with open(args.config, "r") as f:
            if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                config_args = yaml.safe_load(f)
            else:
                config_args = json.load(f)
        # Update defaults with config values, but keep CLI overrides
        for k, v in config_args.items():
            if args_dict.get(k) == parser.get_default(k):  # only override if not set via CLI
                args_dict[k] = v

    # Now you can pass args_dict directly
    train(**args_dict)