def initialize_anchors(anchors_config: list):
    """TODO: Should just make a create model file which prepares all of this

    Extract anchors from the configuration file and initialize them for Yolo

    Args:
        anchors: a 3d python list of containing anchors for each Yolo scale/level
                 see configs.yolov3.model-dn53.yaml for an example

    Returns:
        TODO
    """

    anchors = anchors_config

    # number of anchors per scale
    num_anchors = len(anchors[0])

    # strip away the outer list to make it a 2d python list
    anchors = [anchor for anchor_scale in anchors for anchor in anchor_scale]

    return anchors, num_anchors
