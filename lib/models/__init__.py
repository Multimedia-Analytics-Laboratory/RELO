__all__ = ["build_relo", "build_relo_warmup"]


def build_relo(cfg):
    from .relo import build_relo as _build_relo

    return _build_relo(cfg)


def build_relo_warmup(cfg):
    from .relo_warmup import build_relo_warmup as _build_relo_warmup

    return _build_relo_warmup(cfg)
