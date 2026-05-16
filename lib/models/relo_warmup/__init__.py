__all__ = ["RELOWarmup", "build_relo_warmup"]


def build_relo_warmup(cfg):
    from .relo_warmup import build_relo_warmup as _build_relo_warmup

    return _build_relo_warmup(cfg)


def __getattr__(name):
    if name == "RELOWarmup":
        from .relo_warmup import RELOWarmup

        return RELOWarmup
    raise AttributeError(name)
