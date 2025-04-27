
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """Returns whether distributed training is available and initialized;
    if not, no distributed operations can be used
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """Returns the current process rank (id); 0 for main process"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Whether the current process is the main process"""
    return get_rank() == 0