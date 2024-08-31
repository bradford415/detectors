import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision


def reproducibility(seed: int) -> None:
    """Set the seed for the sources of randomization. This allows for more reproducible results"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_cpu(tensor):
    """Detaches a tensor from the computational graph and transfers it to the CPU.

    detach() works by returning a new tensor that doesn't require a gradient.
    This is very important because PyTorch will continue to try and optimize this tensor, even
    though it is most likely no longer in use; this could cause memory leaks.
    I believe detach() does not have to be callled within torch.no_grad() conext.
    """
    return tensor.detach().cpu()


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = (
                torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            )
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)

        t2 = time.time()

        print("-----------------------------------")
        print("           Preprocess : %f" % (t1 - t0))
        print("      Model Inference : %f" % (t2 - t1))
        print("-----------------------------------")

        return utils.post_processing(img, conf_thresh, nms_thresh, output)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if version.parse(torchvision.__version__) < version.parse('0.7'):
    #     if input.numel() > 0:
    #         return torch.nn.functional.interpolate(
    #             input, size, scale_factor, mode, align_corners
    #         )

    #     output_shape = _output_size(2, input, size, scale_factor)
    #     output_shape = list(input.shape[:-2]) + list(output_shape)
    #     return torchvision.ops.misc._new_empty_tensor(input, output_shape)
    # else:
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
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
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
