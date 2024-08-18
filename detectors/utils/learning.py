
def burnin_schedule(i):
    """Function for the Lambda learning rate scheduler defined by YoloV3 and V4

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