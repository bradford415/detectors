# Detectors
Objection detection model implementations.

## Table of Contents
* [Training a Model](#training-a-model)
* [Resources](#resources)

## Download the COCO dataset
On Linux, the coco dataset can be downloaded by running the bash script. NOTE: this will download the dataset in the current working directory
```bash
bash scripts/bash/download_coco.sh
```

## Environment setup
TODO

## Training a model

### Setting up the configuration files
Change the `root` parameter in `scripts/config/train-coco-config.yaml` to the path of the dataset root. For the coco dataset, this will be the path to the directory containing the `images` and `annotations` dirs.
```bash
dataset:
  root: "/path/to/dataset/root/coco"
```

Begin training by specifying the train configuration file and the desired model configuration file to use 
```bash
python scripts/train.py scripts/configs/train-coco-config.yaml scripts/configs/yolov4/model-base.yaml
```

## Notes
Section which simplifies and clarifies object detection concepts and architecture flows.

### COCO Format
* COCO bbox annotations have the form `[top_left_x, top_left_y, w, h]`

## Resources
### COCO Format
- [COCO object detection format](https://cocodataset.org/#format-data)
  - Explains the coco format of the json annotation file
### YoloV3
- [Really nice implementation of YoloV3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [Explains the YoloV3 model configuration file pretty well](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
