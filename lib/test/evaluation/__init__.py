"""Lightweight public exports for evaluation infrastructure."""

import importlib

__all__ = [
    "BaseDataset",
    "Sequence",
    "SequenceList",
    "Tracker",
    "create_default_eval_local_file",
    "get_dataset",
    "trackerlist",
]


def get_dataset(*args):
    from .datasets import get_dataset as _get_dataset

    return _get_dataset(*args)


def trackerlist(*args, **kwargs):
    from .tracker import trackerlist as _trackerlist

    return _trackerlist(*args, **kwargs)


def create_default_eval_local_file(*args, **kwargs):
    from .environment import create_default_eval_local_file as _create_default_eval_local_file

    return _create_default_eval_local_file(*args, **kwargs)


def __getattr__(name):
    if name in {"BaseDataset", "Sequence", "SequenceList"}:
        module = importlib.import_module(f"{__name__}.data")
    elif name == "Tracker":
        module = importlib.import_module(f"{__name__}.tracker")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
