"""Public exports for the copied training datasets."""

import importlib

_DATASETS = {
    "Got10k": ".got10k",
    "Lasot": ".lasot",
    "MSCOCOSeq": ".coco_seq",
    "TrackingNet": ".tracking_net",
    "VastTrack": ".vasttrack",
}

__all__ = sorted(_DATASETS)


def __getattr__(name):
    module_name = _DATASETS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
