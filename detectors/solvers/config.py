## DO NOT USE, this was a fun experiment I tried but it seemed less flexible than
## putting the solver params in config files
import ml_collections


def yolov3_coco_config():
    """Returns the solver parameters used for the following architecture and dataset

    Detector: YOLOv3
    Backbone: Darknet53
    Dataset: COCO

    Parameters defined in the ResNet paper https://arxiv.org/abs/1512.03385 section 3.4
    """
    config = ml_collections.ConfigDict()

    # Optimizer params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "sgd"
    config.optimizer.params.lr = 0.01
    config.optimizer.params.weight_decay = 5e-4  # 0.0005
    config.optimizer.params.momentum = 0.9

    # Scheduler params
    config.step_lr_on = "epochs"  # step the lr scheduler after n "epochs" or "steps"
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "lambda_lr"

    # labmda_lr function params; see solvers.schedulers.burnin_schedule for details
    config.lr_scheduler.params.function = "burnin_schedule"
    config.lr_scheduler.params.function.burn_in = 1000
    config.lr_scheduler.params.function.steps_threshold = [
        100000,
        130000,
    ]  # ~54 epochs and ~70 epochs (1848 steps per epoch for batch 64)

    return config
