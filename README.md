# Detectors
Objection detection implementations mostly my own learning and practice.

## Table of Contents
* [Training a Model](#training-a-model)
* [Resources](#resources)

## Training a Model
`python scripts/train.py scripts/configs/train-coco-config.yaml scripts/configs/yolov4/model-base.yaml`

## Notes
### COCO Format
* COCO bbox annotations have the form `[top_left_x, top_left_y, w, h]`

## Resources
### COCO Format
- [COCO object detection format](https://cocodataset.org/#format-data)
  - Explains the coco format of the json annotation file
### YoloV3
- [Really nice implementation of YoloV3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
