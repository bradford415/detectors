def burnin_schedule(i):
    """Function for the Lambda learning rate scheduler defined by YoloV3 and V4.
    The learning rate starts very low (close to 0) until it reaches the `burn_in` at 1000 steps
    and increases until it reaches 0.0001


    Mofidied from: https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/train.py#L330
    Parameters defined here: https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/cfg/yolov4.cfg

    Args:
        i: Step number if scheduler.step is called in the dataloader loop; epcoh number if
           scheduler.step is called outside the train/val dataloader
    """
    burn_in = 1000

    steps = [400000, 450000]

    if i < burn_in:
        factor = pow(i / burn_in, 4)
    elif i < steps[0]:
        factor = 1.0
    elif i < steps[1]:
        factor = 0.1
    else:
        factor = 0.01
    return factor


def burnin_schedule_modified(i):
    """Function for the Lambda learning rate scheduler defined by YoloV3 and V4.
    The learning rate starts very low (close to 0) until it reaches the `burn_in` at 1000 steps
    and increases until it reaches the initial learning rate (0.0001)

    After steps[0] number of steps, the learning rate is multiplied by a factor of scales[0]
    After steps[1] number of steps, the learning rate is multiplied by a factor of scales[1]


    Mofidied from: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/b139d49a99b8866d8d4a7cf75a80b1d982abf6f7/pytorchyolo/train.py#L173

    Args:
        i: Step number if scheduler.step is called in the dataloader loop; epcoh number if
           scheduler.step is called outside the train/val dataloader
    """
    burn_in = 1000

    steps = [400000, 450000]
    scales = [0.1, 0.1]

    if i < burn_in:
        factor = i / burn_in
    elif i < steps[0]:
        factor = 1.0
    elif i < steps[1]:
        factor = scales[0]
    else:
        factor = scales[1]
    return factor
