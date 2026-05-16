__all__ = ["RELO", "build_relo"]


def build_relo(cfg):
    from .relo import build_relo as _build_relo

    return _build_relo(cfg)


def __getattr__(name):
    if name == "RELO":
        from .relo import RELO

        return RELO
    raise AttributeError(name)
