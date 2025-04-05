def burnin_schedule_original(i):
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


def burn_in_schedule(current_step, burn_in, steps_thresholds=(100000, 130000)):
    """Function for the Lambda learning rate scheduler defined by YoloV3.

    How the burn_inschedule works:
        - The learning rate starts very low (close to 0) until it reaches the `burn_in` at 1000 steps
           and increases until it reaches the initial learning rate (0.0001)
        -  After steps_thresholds[0] number of steps, the learning rate is multiplied by a factor of scales[0]
        -  After steps_thresholds[1] number of steps, the learning rate is multiplied by a factor of scales[1]

    For example if the initial learning rate is 0.0001, after steps[0] the new lr = 0.0001 * 0.1 = 0.00001
    and after steps[1] the new lr is 0.0001 * 0.01 = 0.000001

    Mofidied from: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/b139d49a99b8866d8d4a7cf75a80b1d982abf6f7/pytorchyolo/train.py#L173

    Args:
        current_step: Step number if scheduler.step is called in the dataloader loop; epcoh number if
                      scheduler.step is called outside the train/val dataloader
        steps_thresholds: two step thresholds to multiply the learning rate by a factor once
                          the steps reach the threshold; default is about ~54 epochs and ~70 epochs (1848 steps per epoch for batch 64)
        burn_in: number of steps to linearly increase the learning rate from 0 to the initial learning rate
    """

    # Default steps in the yolo config batch_size of 64 (original yolov3 implementation)
    # burn_in = 1000
    # steps = [400000, 450000] # ~261 epochs and ~243 epochs
    # scales = [0.1, 0.01]

    # Default steps in the yolo config batch_size of 64 (original yolov3 implementation)
    burn_in = 1000
    scales = [0.1, 0.01]

    # My logic for choosing the step intervals based on the papers batch size of 64:
    # 64*400000 = 25,600,000 samples -> 25600000 / 16 = 1,600,000 (~216 epochs) therefore we should reduce the lr after 1.6m steps
    # 64*450000 = 28,800,000 samples -> 28800000 / 16 = 1,800,000 ()
    # batch_size of 16
    # burn_in = 4000
    # steps = [1600000, 1800000]
    # scales = [0.1, 0.01]

    if current_step < burn_in:
        factor = current_step / burn_in
    elif current_step < steps_thresholds[0]:
        factor = 1.0
    elif current_step < steps_thresholds[1]:
        factor = scales[0]
    else:
        factor = scales[1]
    return factor
