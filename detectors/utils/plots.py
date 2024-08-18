import matplotlib.pyplot as plt
import torch


def visualize_img_tensors(img_tensors):
    img_tensors = img_tensors.to("cpu")

    fig, ax = plt.subplots(1, 1)
    for index, image in enumerate(img_tensors):
        breakpoint()
        image = image.permute(1, 2, 0)
        ax.imshow(image.to(dtype=torch.uint8), vmin=0, vmax=255)
        fig.savefig(f"output/temp-images/image_tensor_{index}.png")
