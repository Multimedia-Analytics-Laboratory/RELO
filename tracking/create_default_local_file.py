import argparse
import os
import _init_paths
from lib.train.admin import create_default_train_local_file
from lib.test.evaluation import create_default_eval_local_file


def parse_args():
    parser = argparse.ArgumentParser(description='Create default local path files')
    parser.add_argument("--workspace_dir", type=str, required=True)  # workspace dir
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    workspace_dir = os.path.realpath(args.workspace_dir)
    data_dir = os.path.realpath(args.data_dir)
    save_dir = os.path.realpath(args.save_dir)
    create_default_train_local_file(workspace_dir, data_dir)
    create_default_eval_local_file(workspace_dir, data_dir, save_dir)
