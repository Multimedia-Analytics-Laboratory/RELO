from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.relo import build_relo
from lib.models.relo_warmup import build_relo_warmup
from lib.train.actors import RELOActor, RELOWarmupActor
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler, update_settings
from lib.train.trainers import LTRTrainer
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


@dataclass(frozen=True)
class TrainingRoute:
    builder: Callable
    actor_cls: type
    find_unused_parameters: bool


_PUBLIC_TRAINING_ROUTES = {
    "relo_warmup": TrainingRoute(
        builder=build_relo_warmup,
        actor_cls=RELOWarmupActor,
        find_unused_parameters=False,
    ),
    "relo": TrainingRoute(
        builder=build_relo,
        actor_cls=RELOActor,
        find_unused_parameters=True,
    ),
}

PUBLIC_SCRIPT_NAMES = frozenset(_PUBLIC_TRAINING_ROUTES)


def init_seeds(seed):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_training_route(script_name):
    try:
        return _PUBLIC_TRAINING_ROUTES[script_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PUBLIC_SCRIPT_NAMES))
        raise ValueError(
            f"Unsupported public training script '{script_name}'. "
            f"Allowed script names: {allowed}."
        ) from exc


def config_file_path(script_name, config_name):
    resolve_training_route(script_name)
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return os.path.join(prj_dir, "experiments", script_name, f"{config_name}.yaml")


def load_training_config(script_name, config_name, cfg_file=None):
    resolve_training_route(script_name)
    cfg_file = cfg_file or config_file_path(script_name, config_name)
    if not os.path.exists(cfg_file):
        raise ValueError(f"{cfg_file} doesn't exist.")

    config_module = importlib.import_module(f"lib.config.{script_name}.config")
    config_module.update_config_from_file(cfg_file)
    return config_module.cfg


def build_actor_losses(script_name, cfg):
    resolve_training_route(script_name)

    if script_name == "relo":
        objective = {"giou": giou_loss}
        loss_weight = {}
        loss_weight["rl"] = cfg.TRAIN.RL_WEIGHT
    else:
        objective = {
            "giou": giou_loss,
            "l1": l1_loss,
            "focal": FocalLoss(),
        }
        loss_weight = {
            "giou": cfg.TRAIN.GIOU_WEIGHT,
            "l1": cfg.TRAIN.L1_WEIGHT,
            "focal": getattr(cfg.TRAIN, "FOCAL_WEIGHT", 1.0),
        }

    return objective, loss_weight


def build_network(script_name, cfg):
    route = resolve_training_route(script_name)
    return route.builder(cfg)


def build_actor(script_name, net, cfg, settings):
    route = resolve_training_route(script_name)
    objective, loss_weight = build_actor_losses(script_name, cfg)
    return route.actor_cls(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)


def _current_dist_rank():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process(settings):
    if hasattr(settings, "is_main_process"):
        return bool(settings.is_main_process)
    if hasattr(settings, "global_rank"):
        return settings.global_rank == 0
    return _current_dist_rank() == 0 and getattr(settings, "local_rank", -1) in [-1, 0]


def _print_config(cfg):
    print("New configuration is shown below.")
    for key in cfg.keys():
        print("%s configuration:" % key, cfg[key])
        print("\n")


def _resolve_device(local_rank):
    if torch.cuda.is_available():
        cuda_index = local_rank if local_rank != -1 else 0
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def _move_and_wrap_network(net, settings, route):
    local_rank = settings.local_rank
    device = _resolve_device(local_rank)
    net.to(device)
    settings.device = device

    distributed = getattr(settings, "distributed", local_rank != -1)
    if not distributed:
        return net
    if local_rank == -1:
        raise RuntimeError(
            "Distributed training requested but no LOCAL_RANK was resolved. "
            "Launch with torchrun or pass --local-rank/--local_rank."
        )

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed training requested but torch.distributed is not initialized. "
            "Launch with torchrun so LOCAL_RANK/RANK/WORLD_SIZE are set."
        )

    ddp_kwargs = {
        "broadcast_buffers": False,
        "find_unused_parameters": route.find_unused_parameters,
    }
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [local_rank]

    net = DDP(net, **ddp_kwargs)
    for buffer in net.buffers():
        dist.broadcast(buffer, src=0)
    return net


def _checkpoint_search_roots(settings):
    roots = []
    for root in (
        getattr(settings, "save_dir", None),
        getattr(getattr(settings, "env", None), "workspace_dir", None),
        os.getcwd(),
    ):
        if root:
            abs_root = os.path.abspath(os.path.expanduser(root))
            if abs_root not in roots:
                roots.append(abs_root)
    return roots


def resolve_previous_checkpoint_path(checkpoint, settings):
    checkpoint = os.path.expanduser(checkpoint)
    if os.path.isabs(checkpoint):
        return os.path.abspath(checkpoint)

    candidates = []
    normalized = checkpoint.replace("\\", "/")
    for root in _checkpoint_search_roots(settings):
        if normalized == "checkpoints" or normalized.startswith("checkpoints/"):
            candidates.append(os.path.join(root, checkpoint))
        else:
            candidates.append(os.path.join(root, "checkpoints", checkpoint))
            candidates.append(os.path.join(root, checkpoint))

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(candidates[0]) if candidates else os.path.abspath(checkpoint)


def validate_previous_checkpoint(script_name, cfg, settings):
    if script_name != "relo":
        return None

    checkpoint = getattr(cfg.TRAIN, "PRV_CKPT", None)
    if not checkpoint:
        raise FileNotFoundError("Previous checkpoint is required for relo training: TRAIN.PRV_CKPT is empty.")

    resolved = resolve_previous_checkpoint_path(checkpoint, settings)
    if not os.path.exists(resolved):
        roots = ", ".join(_checkpoint_search_roots(settings)) or os.getcwd()
        raise FileNotFoundError(
            f"Previous checkpoint for relo training was not found: {checkpoint} "
            f"(resolved to {resolved}; searched from {roots})."
        )
    return resolved


def run(settings):
    settings.description = "Public RELO training"
    route = resolve_training_route(settings.script_name)

    cfg = load_training_config(settings.script_name, settings.config_name, settings.cfg_file)
    if is_main_process(settings):
        _print_config(cfg)

    update_settings(settings, cfg)

    log_base_dir = settings.save_dir or getattr(settings.env, "workspace_dir", None) or os.getcwd()
    log_dir = os.path.join(log_base_dir, "logs")
    if is_main_process(settings):
        os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type != "tracking":
        raise ValueError("Unsupported DATA.LOADER for public RELO training: %s" % loader_type)
    loader_train = build_dataloaders(cfg, settings)

    net = build_network(settings.script_name, cfg)
    net = _move_and_wrap_network(net, settings, route)

    actor = build_actor(settings.script_name, net, cfg, settings)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    if settings.script_name == "relo":
        load_previous_ckpt = validate_previous_checkpoint(settings.script_name, cfg, settings)
        policy_load_decoder = cfg.MODEL.POLICY.LOAD_DECODER
    else:
        load_previous_ckpt = None
        policy_load_decoder = False

    trainer.train(
        cfg.TRAIN.EPOCH,
        load_latest=True,
        fail_safe=False,
        load_previous_ckpt=load_previous_ckpt,
        policy_load_decoder=policy_load_decoder,
    )


def run_training(
    script_name,
    config_name,
    cudnn_benchmark=True,
    local_rank=-1,
    save_dir=None,
    base_seed=None,
    use_lmdb=False,
    distributed=False,
    global_rank=0,
    world_size=1,
    is_main_process=True,
):
    """Run a public RELO training script."""
    import cv2 as cv
    import lib.train.admin.settings as ws_settings

    resolve_training_route(script_name)
    local_rank = -1 if local_rank is None else int(local_rank)

    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if base_seed is not None:
        init_seeds(base_seed + local_rank if local_rank != -1 else base_seed)

    print("script_name: {}.py  config_name: {}.yaml".format(script_name, config_name))

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = "train/{}/{}".format(script_name, config_name)
    settings.local_rank = local_rank
    settings.distributed = bool(distributed)
    settings.global_rank = int(global_rank)
    settings.world_size = int(world_size)
    settings.is_main_process = bool(is_main_process)
    settings.save_dir = os.path.abspath(save_dir) if save_dir is not None else None
    settings.use_lmdb = bool(use_lmdb)
    settings.cfg_file = config_file_path(script_name, config_name)

    return run(settings)
