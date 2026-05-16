import argparse
import os
from dataclasses import dataclass

try:
    from . import _init_paths
except ImportError:
    import _init_paths


@dataclass(frozen=True)
class DistributedState:
    local_rank: int
    rank: int
    world_size: int
    distributed: bool
    is_main_process: bool


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
    """Run the public train script.
    args:
        script_name: Name of experiment in the "experiments/" folder.
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    from lib.train.train_script import run_training as run_training_entrypoint

    return run_training_entrypoint(
        script_name,
        config_name,
        cudnn_benchmark=cudnn_benchmark,
        local_rank=local_rank,
        save_dir=save_dir,
        base_seed=base_seed,
        use_lmdb=use_lmdb,
        distributed=distributed,
        global_rank=global_rank,
        world_size=world_size,
        is_main_process=is_main_process,
    )


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run public RELO training. For distributed jobs, launch with torchrun so LOCAL_RANK is set."
    )
    parser.add_argument('--script', type=str, required=True, help='Name of the train script.')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file.")
    parser.add_argument(
        '--cudnn_benchmark',
        type=str_to_bool,
        default=True,
        help='Set cudnn benchmark on or off (default is on).',
    )
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        dest='local_rank',
        default=None,
        type=int,
        help='Backward-compatible local-rank override; torchrun users should rely on LOCAL_RANK.',
    )
    parser.add_argument('--save_dir', type=str, help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=None, help='seed for random numbers')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    return parser


def resolve_local_rank(cli_local_rank=-1):
    """Resolve local rank for torchrun, with a legacy CLI override."""
    return resolve_distributed_state(cli_local_rank).local_rank


def _env_int(name, default, env=None):
    env = os.environ if env is None else env
    value = env.get(name)
    if value is None:
        return default
    return int(value)


def resolve_distributed_state(cli_local_rank=-1, env=None):
    env = os.environ if env is None else env
    cli_has_local_rank = cli_local_rank is not None and cli_local_rank != -1
    env_has_local_rank = env.get("LOCAL_RANK") is not None
    env_has_rank = "RANK" in env
    env_has_world_size = "WORLD_SIZE" in env

    if cli_has_local_rank:
        local_rank = int(cli_local_rank)
    elif env_has_local_rank:
        local_rank = int(env["LOCAL_RANK"])
    else:
        local_rank = -1

    rank = _env_int("RANK", 0, env)
    world_size = _env_int("WORLD_SIZE", 1, env)
    distributed_env = env_has_local_rank or env_has_rank or env_has_world_size

    if (env_has_rank or env_has_world_size) and local_rank == -1:
        raise RuntimeError(
            "Distributed environment detected from RANK/WORLD_SIZE, but no LOCAL_RANK "
            "or --local-rank/--local_rank was provided. Launch with torchrun or pass a local rank."
        )

    distributed = distributed_env or cli_has_local_rank
    return DistributedState(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        is_main_process=rank == 0,
    )


def should_init_distributed(distributed_state):
    if isinstance(distributed_state, DistributedState):
        return distributed_state.distributed
    return resolve_distributed_state(distributed_state).distributed


def init_distributed_if_needed(distributed_state):
    if not isinstance(distributed_state, DistributedState):
        distributed_state = resolve_distributed_state(distributed_state)

    if not should_init_distributed(distributed_state):
        return False
    if distributed_state.local_rank == -1:
        raise RuntimeError("Distributed training requires a resolved LOCAL_RANK.")

    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(distributed_state.local_rank)
    return True


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    distributed_state = resolve_distributed_state(args.local_rank)
    initialized_distributed = init_distributed_if_needed(distributed_state)

    if not initialized_distributed:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    print("local_rank:", distributed_state.local_rank)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=distributed_state.local_rank, save_dir=args.save_dir, base_seed=args.seed,
                 use_lmdb=args.use_lmdb, distributed=distributed_state.distributed,
                 global_rank=distributed_state.rank, world_size=distributed_state.world_size,
                 is_main_process=distributed_state.is_main_process)


if __name__ == '__main__':
    main()
