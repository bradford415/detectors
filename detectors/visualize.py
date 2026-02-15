import random
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullLocator
from PIL import Image

from detectors.data.transforms.transforms import Unnormalize
from detectors.utils.box_ops import rescale_boxes
from detectors.utils.misc import to_cpu

matplotlib.use("Agg")


def visualize_norm_img_tensors(
    img_tensors: torch.Tensor,
    targets: torch.Tensor,
    annotations: tuple[dict],
    step: int,
    classes: list[str],
    output_dir: Path,
):
    """Visualizes the boxes of augmented images just before the input of the model; this helps
    manually verify the data augmentation on the images and labels is accurate

    Args:
        img_tensors: tensor of normalized images (b, c, h, w)
        targets: list of dicts containing at least the image bboxes; bboxes format (cx, cy, w, h)
        classes: list of unique class names by label index
        output_dir: Path to save the outputs
    """
    # assert img_tensors.shape[0] == targets.shape[0]

    un_norm = Unnormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img_tensors = to_cpu(img_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)

    unique_classes = np.unique(np.array(targets[:, 1]))
    num_unique_classes = len(unique_classes)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)

    for img_index, image in enumerate(img_tensors):
        fig, ax = plt.subplots(1)

        img_h, img_w = image.shape[1:]

        image = un_norm(image)
        image = image.permute(1, 2, 0)
        # ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        ax.imshow(image)

        for img_idx, label, cx, cy, w, h in targets[targets[:, 0] == img_index]:
            # box coords are normalize [0,1] so we need to scale them to the input size
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            tl_x = cx - w // 2
            tl_y = cy - h // 2

            color = bbox_colors[int(np.where(unique_classes == int(label))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (int(tl_x), int(tl_y)),
                int(w),
                int(h),
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            # Add the bbox to the plot
            # TODO: need to figure out if I need to clip these boxes because sometimes the figure is too large
            # Might be in rescale_boxes() in github code
            ax.add_patch(bbox)
            plt.text(
                tl_x,
                tl_y,
                s=f"{classes[int(label)]}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.axis("off")
        fig.savefig(
            f"{output_dir}/{int(step)}-augmented.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()

    # Plot the original image to compare against the augmented images
    for img_idx, ann in enumerate(annotations):
        img_path = ann["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))
        fig, ax = plt.subplots(1)

        ax.imshow(image)
        plt.axis("off")
        fig.savefig(
            f"{output_dir}/{int(step)}_original.png",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def visualize_dataloader(
    img_tensors: torch.Tensor,
    targets: torch.Tensor,
    annotations: tuple[dict],
    step: int,
    classes: list[str],
    output_dir: Path,
):
    """
    NOTE: this was an attempt at plotting them side by side but did not look great

    Visualizes the boxes of augmented images just before the input of the model; this helps
    manually verify the data augmentation on the images and labels is accurate

    Args:
        img_tensors: tensor of normalized images (b, c, h, w)
        targets: list of dicts containing at least the image bboxes; bboxes format (cx, cy, w, h)
        classes: list of unique class names by label index
        output_dir: Path to save the outputs
    """
    # assert img_tensors.shape[0] == targets.shape[0]

    un_norm = Unnormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img_tensors = to_cpu(img_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)

    unique_classes = np.unique(np.array(targets[:, 1]))
    num_unique_classes = len(unique_classes)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)

    for img_index, (image, ann) in enumerate(zip(img_tensors, annotations)):
        fig, ax = plt.subplots(1, 2)

        img_h, img_w = image.shape[1:]

        image = un_norm(image)
        image = image.permute(1, 2, 0)
        # ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        ax[0].imshow(image)

        for img_idx, label, cx, cy, w, h in targets[targets[:, 0] == img_index]:
            # box coords are normalize [0,1] so we need to scale them to the input size
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            tl_x = cx - w // 2
            tl_y = cy - h // 2

            if tl_x < -1000.0 or tl_y < -1000.0 or w > 10000.0 or h > 10000.0:
                continue

            color = bbox_colors[int(np.where(unique_classes == int(label))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (tl_x, tl_y), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            # Add the bbox to the plot
            # TODO: need to figure out if I need to clip these boxes because sometimes the figure is too large
            # Might be in rescale_boxes() in github code
            ax[0].add_patch(bbox)
            ax[0].text(
                tl_x,
                tl_y,
                s=f"{classes[int(label)]}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        ax[0].axis("off")

        # Plot the original image to compare against the augmented images
        img_path = ann["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))

        ax[1].imshow(image)
        ax[1].axis("off")
        fig.savefig(
            f"{output_dir}/augmented_{int(step)}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def plot_test_detections(
    img_detections: dict,
    conf_threshold: float,
    classes: list[str],
    plot_n_images: int,
    output_dir: Path,
):
    """visualize the predictions from the test script;

    NOTE: expects the image file path in the `img_detections` dict

    Args:
        img_detections: dictionary of image detections from the test set, sorted by decreasing
                        confidence threshold; contains the image_id as the key and the following
                        values:
                          image_path: the path to the original images used to load and plot the
                                      predictions on (num_select,)
                          scores: the confidence scores for each detection
                          boxes: the ABSOLUTE bbox predictions in XYXY format; these have already
                                 been scaled back to the original image size through the postprocessor
                                 (num_select, 4)
                          labels: the predicted class that each box belongs to (num_select,)
        conf_threshold: the user-defined confidence threshold to filter
        classes: list of unique class labels representing the name of each class; used to get the label
                 name from the predicted class_id
        plot_n_images: number of images to plot detections on
        output_dir: path to the directory to save the detections
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_id, detects) in enumerate(img_detections.items()):
        # extract the conf_scores, bbox preds, class_preds, filtered by the confidence threshold
        keep = detects["scores"] >= conf_threshold
        conf_scores = detects["scores"][keep]
        box_preds = detects["boxes"][keep]
        class_preds = detects["labels"][keep]
        img_path = detects["image_path"]

        assert conf_scores.shape[0] == box_preds.shape[0] == class_preds.shape[0]

        filtered_detects = torch.cat(
            [box_preds, conf_scores[:, None], class_preds[:, None]], dim=1
        )

        save_name = output_dir / f"image_detections_{idx}.jpg"
        plot_detections(img_path, filtered_detects, classes, save_name)

        if (idx + 1) == plot_n_images:
            break


def plot_all_detections(
    img_detections: list[dict],
    conf_threshold: float,
    classes: dict[int, str],
    plot_n_images: int,
    output_dir: Path,
    img_id: Optional[int] = None,
):
    """visualize the predictions

    Args:
        img_detections: dictionary of image detections from the test set, sorted by decreasing
                        confidence threshold; contains the image_id as the key and the following
                        values:
                          image_path: the path to the original images used to load and plot the
                                      predictions on (num_select,)
                          scores: the confidence scores for each detection
                          boxes: the ABSOLUTE bbox predictions in XYXY format; these have already
                                 been scaled back to the original image size through the postprocessor
                                 (num_select, 4)
                          labels: the predicted class that each box belongs to (num_select,)
        conf_threshold: the user-defined confidence threshold to filter
        classes: list of unique class labels representing the name of each class; used to get the label
                 name from the predicted class_id
        plot_n_images: number of images to plot detections on
        output_dir: path to the directory to save the detections
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, detects in enumerate(img_detections):
        # extract the conf_scores, bbox preds, class_preds, filtered by the confidence threshold
        keep = detects["scores"] >= conf_threshold
        conf_scores = detects["scores"][keep]
        box_preds = detects["boxes"][keep]
        class_preds = detects["labels"][keep]
        img_path = detects["image_path"]

        assert conf_scores.shape[0] == box_preds.shape[0] == class_preds.shape[0]

        filtered_detects = torch.cat(
            [box_preds, conf_scores[:, None], class_preds[:, None]], dim=1
        )

        if img_id is not None:
            save_name = output_dir / f"image_detections_{img_id}.jpg"
        else:
            save_name = output_dir / f"image_detections_{idx}.jpg"
        plot_detections(img_path, filtered_detects, classes, save_name)

        if (idx + 1) == plot_n_images:
            break


def plot_abs_detections(image):
    """plot detections on an image given an image path and absolute bbox predictions"""


def plot_detections(
    image_path: str,
    detections: torch.Tensor,
    classes: dict[int, str],
    save_name: str,
    img_size: Optional[int] = None,
):
    """Visualizes the augmented images just before the input of the model; this helps
    manually verify the data augmentation on the images and labels is accurate

    Args:
        image_path: the image path of the image being detected
        detections: detections after non-max suppression (num_detections, 6);
                    detected boxes should be (tl_x, tl_y, br_x, br_y, conf, cls)
        targets: Dictionaries containing at least the ground truth bboxes and label for each
                 object; bboxes should be (tl_x)each element of the list is an image's labels
        classes: ditionary of unique class ids to label name
        output_dir: Path to save the outputs
        img_size: image input size to the model; this should be the size the detedtions
                  were scaled to,
    """
    img_pil = Image.open(image_path).convert("RGB")

    # TODO: might need to remove this since I'm now keeping the aspect ratio when resizing
    if img_size is not None:
        img_pil = img_pil.resize(size=(img_size, img_size))

    img = np.array(img_pil)

    fig, ax = plt.subplots(1)

    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.numpy()

    detections[:, 0]
    labels = detections[:, -1].astype(np.uint8)
    unique_classes = np.unique(np.array(labels))
    num_unique_classes = len(unique_classes)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)
    for tl_x, tl_y, br_x, br_y, conf, cls_pred in detections:

        if tl_x < -1000.0 or tl_y < -1000.0 or br_x > 10000.0 or br_y > 10000.0:
            continue

        box_width = br_x - tl_x
        box_height = br_y - tl_y

        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]

        # Create a Rectangle patch
        bbox = patches.Rectangle(
            (tl_x, tl_y),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        # Add the bbox to the plot
        ax.add_patch(bbox)
        ax.text(
            tl_x,
            tl_y,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )

    ax.axis("off")
    fig.savefig(save_name, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


# def plot_loss(train_loss: list[float], val_loss: list[float]=None, save_dir: str):
def plot_loss(train_loss: list[float], val_loss: list[float], save_dir: str):
    """Plots the total loss"""
    save_name = Path(save_dir) / "total_loss.jpg"

    x = np.arange(len(train_loss)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, train_loss)
    ax.plot(x, val_loss)

    plt.legend(["train loss", "val loss"])
    plt.title("total loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def plot_mAP(val_mAP: list[float], save_dir: str):
    """Plots the validation mAP per epoch"""
    save_name = Path(save_dir) / "mAP_curve.jpg"

    x = np.arange(len(val_mAP)) + 1
    fig, ax = plt.subplots(1)
    ax.plot(x, val_mAP)

    plt.title("validation mAP per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")

    fig.savefig(save_name, bbox_inches="tight")
    plt.close()


def visualize_batch(
    self, dataloader: torch.utils.data.DataLoader, split: str, class_names: List[str]
):
    """Visualize a batch of images after data augmentation; sthis helps manually verify
    the data augmentations are working as intended on the images and boxes

    Args:
        dataloader: Train or val dataloader
        split: "train" or "val"
        class_names: List of class names in the ontology
    """
    valid_splits = {"train", "val"}
    if split not in valid_splits:
        raise ValueError("split must either be in valid_splits")

    dataiter = iter(dataloader)
    samples, targets, annotations = next(dataiter)
    visualize_norm_img_tensors(
        samples,
        targets,
        annotations,
        class_names,
        self.output_dir / "aug" / f"{split}-images",
    )
