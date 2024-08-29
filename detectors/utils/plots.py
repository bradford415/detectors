import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import numpy as np

from detectors.data.transforms import UnNormalize


def visualize_norm_img_tensors(img_tensors: torch.Tensor, targets, classes, output_dir: Path):
    """TODO"""

    un_norm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # assert torch.max(img_tensors) <= 1.0

    img_tensors = img_tensors.to("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = []
    for target in targets:
        labels += target["labels"]
    unique_classes = np.unique(np.array(labels))
    num_unique_classes = len(unique_classes)
    
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, num_unique_classes)]
    bbox_colors = random.sample(colors, num_unique_classes)

    for index, image in enumerate(img_tensors):
        fig, ax = plt.subplots(1, 1)
        
        image = un_norm(image)
        image = image.permute(1, 2, 0)
        # ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        ax.imshow(image)
        breakpoint()        
        for label_num, (x1, y1, w, h) in zip(targets[index]["labels"], targets[index]["boxes"]):

            tl_x = x1 - w // 2
            tl_y = y1 - h // 2

            color = bbox_colors[int(np.where(unique_classes == int(label_num))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((tl_x, tl_y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            plt.text(
            tl_x,
            tl_y,
            s=f"{classes[int(label_num)]}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

        #plt.gca().xaxis.set_major_locator(NullLocator())
        #plt.gca().yaxis.set_major_locator(NullLocator())
        fig.savefig(f"{output_dir}/image_tensor_{index}.png")
        plt.close()
    
    
    
def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
