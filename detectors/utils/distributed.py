import logging
import os

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


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


def get_world_size():
    """returns the number of processes (GPUs) being used in the distributed training setup"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Whether the current process is the main process"""
    return get_rank() == 0


def init_distributed_mode(backend: str = "nccl", dist_url: str = "env://"):
    """Initalize torch distributed mode (multi-gpu training); each process calls this method

    Sets the gpu to use for each process and initalizes the process group

    Extracts the following information:
        world_size: the total number of processes (GPUs) across all nodes;
                    (e.g., 2 nodes w/ 8 gpus each -> world_size= 2 * 8 = 16)
        rank: index of the current process (GPU) globally across all nodes [0, world_size-1];
              (e.g., 2 nodes w/ 8 gpus each the rank could be 0-15)
        local_rank: index of the current process (GPU) on the local machine/node;
                    (e.g., 2 nodes w/ 8 gpus each, if on the 1st node the local rank could be [0,7])

    Args:
        backend: the communication backend for distributed training; for cuda "nccl" should be used
        dist_url: the distributed URL that specifies how the different processes in a distributed
                  pytorch job discover and communicate with each other during initialization; pytorch
                  default is "env://"
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        log.info("using torch distributed mode")

        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        log.info(
            "world size: %d, global rank: %d, local rank: %d",
            world_size,
            global_rank,
            local_rank,
        )
    else:
        log.info("Not using distributed mode")

        world_size = 1
        global_rank = 0
        local_rank = 0
        return world_size, global_rank, local_rank

    # set the gpu number to use for the current process/gpu;
    # e.g., if process/gpu 2 is running this statement, this process will be assigned gpu 2 because
    #       the local_rank=2
    torch.cuda.set_device(local_rank)

    log.info("| distributed init (rank %d): %s", global_rank, dist_url)

    # initialize the distributed communication group for torch distributed mode;
    # sets up the communcation between processes and initalizes the communication backend;
    # each process/gpu calls this once
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=global_rank
    )

    # blocks all processes (in the process group) from continuing until every process has
    # reached this barrier; it ensures all processes are synchronized at a this point in the code
    torch.distributed.barrier()

    ######## start here - verify this is finished and continue (need to call this in train.py i think)
