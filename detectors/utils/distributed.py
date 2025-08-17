import logging
import os
import pickle
from typing import Optional

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


def get_global_rank():
    """Returns the current global process rank (id); 0 for main process"""
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
    return get_global_rank() == 0


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

    Returns:
        1. world size
        2. global_rank (proccess/gpu id out of all processes/gpus across all nodes)
        3. local_rank id (process/gpu id from only the current node)
        4. whether distributed training is being used (torchrun)
    """
    distruted_mode = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        log.info("using torch distributed mode")
        distruted_mode = True

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
        return world_size, global_rank, local_rank, distruted_mode

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

    return world_size, global_rank, local_rank, distruted_mode


def reduce_dict(
    input_dict: dict[str, torch.tensor],
    average: bool = True,
):
    """Average the dictionaries values across all processes

    Args:
        input_dict: a dictionary of tensor values, typically representing the loss components
        average: whether to average the values across processes; if False, only sum values

    Returns:
        a dictionary with values averages across all processes
    """
    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        # sort the keys so that they are consistent across processes
        for key in sorted(input_dict.keys()):
            names.append(key)
            values.append(input_dict[key])

        # stack the tensor values so we can perform all_reduce() in one call
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        # average by the number of processes to get an average loss for every gpu
        if average:
            values /= world_size

        # create the new averaged dict
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def synchronize_loss_between_processes(count: int, loss_dict: dict):
    """Average the loss componenets in a dict across all_processes

    Can be used for computing the average loss for an epoch

    Args:
        count: the number of times the loss componenets were summed
        loss_dict: a dictionary of loss components; typically a running sum so we can average

    Returns:
        a dictionary of averaged loss components
    """
    if not is_dist_avail_and_initialized():
        return loss_dict

    # TODO: verify this makes sense; I honestly don't know but I tried (:
    reduced_dict = {}
    for loss_type, loss_val in loss_dict.items():
        # sum the running loss & num_steps across GPUs
        t = torch.tensor([count, loss_val], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)

        # average the loss across all the steps and processes
        t = t.tolist()
        reduced_dict[loss_type] = int(t[0]) / t[1]

    return reduced_dict


def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors)

    Useful for arbitrary python objects of different types and sizes that cannot
    necessarily be converted to tensors; this function serializes the data
    to a byte string, converts it to a ByteTensor, and then gathers the tensors
    from all ranks; the data is then deserialized back to the original python object

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor; data can be of arbitrary shape and types which
    # likely cannot be converted to a tensor, so we can first serialize it to a byte string
    # (which nearly anything type or shape can be converted to this), then we convert the
    # byte string to a ByteTensor which can be used in torch distributed operations
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # find out each ranks tensor size since it could be of different sizes; we'll create an empty
    # tensor list padded to the maximum size since torch.all_gather does not support different sizes
    # the tensor to transmit to other ranks
    local_size = torch.tensor([tensor.numel()], device="cuda")

    # tensor to receive the size of each rank's tensor
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]

    # gather the size of each rank's tensor
    dist.all_gather(size_list, local_size)

    # find the maximum size of the tensors across all ranks
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)

    # a list of tensor values from all ranks
    dist.all_gather(tensor_list, tensor)

    # deserialize the tensors by
    #   1. removing the padding using the original sizes
    #   2. converting the byte tensors back to the original python object
    #   3. data_list is now a list of each proccess' original data
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
