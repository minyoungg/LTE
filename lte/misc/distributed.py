import os
import torch
import torch.distributed as dist


def num_visible_devices():
    return torch.cuda.device_count()
    

def reduce(tensor, reduction):
    """DDP reduction across devices"""
    if not is_distributed():
        return tensor
    tensor = torch.tensor(tensor, device="cuda")
    dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    dist.barrier()
    if reduction == "mean":
        tensor /= dist.get_world_size()
    elif reduction == "sum":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return tensor


def is_distributed():
    return dist.is_initialized() and dist.is_available()


def local_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def device():
    return torch.device("cuda", dist.get_rank())


def is_main_process():
    return local_rank() == 0


def init_distributed_mode(args):
    """Initialize distributed mode"""

    def setup_for_distributed(is_master):
        """Disables printing when not in master process"""
        import builtins as __builtin__

        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
