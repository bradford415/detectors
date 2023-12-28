from pycocotools.coco import COCO


def explore_coco(coco_annotation: COCO):

    print("\nDisplaying COCO information:")
    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")

    img_ids = coco_annotation.getImgIds()
    print(f"Number of Images: {len(img_ids)}")
