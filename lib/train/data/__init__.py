"""Public exports for the copied training data infrastructure."""

__all__ = [
    "LTRLoader",
    "default_image_loader",
    "jpeg4py_loader",
    "jpeg4py_loader_w_failsafe",
    "opencv_loader",
]


def __getattr__(name):
    if name == "LTRLoader":
        from .loader import LTRLoader

        return LTRLoader

    if name in {
        "default_image_loader",
        "jpeg4py_loader",
        "jpeg4py_loader_w_failsafe",
        "opencv_loader",
    }:
        from . import image_loader

        return getattr(image_loader, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
