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

__linux__
TODO

__mac__
```bash
python -m venv .venv
source .venv/bin/activate
make install_reqs
```

## Training a model
This project is designed to use the configuration specified in `scripts/configs/`, but for ease of use the CLI arguments specified below will overwrite the main default config parameters for quick setup.

## Single-GPU training
### Training from scratch (or backbone pretrained from ImageNet weights)
```bash
# yolov3 with a darknet53 backbone
python scripts/train.py --dataset_root "/path/to/coco"

# dino detr
python scripts/train.py configs/train-coco-dino-rn50.yaml configs/dino/dino-rn50.yaml
```

### Training from scratch with a pre-trained backbone
```bash
python scripts/train.py --dataset_root "/mnt/d/datasets/coco" --backbone_weights "/path/to/backbone_weights.pt"
```

### Resume training from a checkpoint
```bash
python scripts/train.py --dataset_root "/mnt/d/datasets/coco" --checkpoint_path "/path/to/checkpoint_weights.pt"
```

## Multi-GPU Training
```bash
# distributed training dino w/ a resnet50 backbone
torchrun --nproc_per_node=<num_gpus> scripts/train.py configs/train-coco-dino-rn50.yaml configs/dino/dino-rn50.yaml
```

## Inferencing
TODO

## Results
| Detector / Backbone | Pretrained Backbone | Dataset  | Best mAP / Epoch |
|---------------------|---------------------|----------|------------------|
| YoloV3 / DarkNet53  | Scratch             | COCO     | 39.2% / 59       |
| YoloV3 / DarkNet53  | ImageNet            | COCO     | 40.3% / 46       |


## Troubleshooting
* [Debugging MultiScaleDeformableAttention installation](detectors/models/ops)

## Notes
Section which simplifies and clarifies object detection concepts and architecture flows.

### bbox formats
* COCO bbox  - `XYWH` -  `[top_left_x, top_left_y, w, h]`

## Resources
### COCO Format
- [COCO object detection format](https://cocodataset.org/#format-data)
  - Explains the coco format of the json annotation file
### YoloV3
- [Really nice implementation of YoloV3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [Explains the YoloV3 model configuration file pretty well](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
