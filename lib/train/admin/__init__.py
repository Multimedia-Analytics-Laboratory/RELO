"""Lightweight exports for training administration helpers."""

__all__ = [
    "AverageMeter",
    "MultiGPU",
    "StatValue",
    "TensorboardWriter",
    "create_default_train_local_file",
    "env_settings",
]


def __getattr__(name):
    if name == "MultiGPU":
        from .multigpu import MultiGPU

        return MultiGPU
    if name in {"env_settings", "create_default_train_local_file"}:
        from . import environment

        return getattr(environment, name)
    if name in {"AverageMeter", "StatValue"}:
        from . import stats

        return getattr(stats, name)
    if name == "TensorboardWriter":
        from .tensorboard import TensorboardWriter

        return TensorboardWriter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
