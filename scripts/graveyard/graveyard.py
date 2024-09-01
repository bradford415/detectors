# This file is full of code snippets that I tried and either did not work or
# did not decide to put in the main codebase; these could be valuable in
# the future so I'll keep them here

########################### COCO EVALUATOR ################################

# tracemalloc.start()
# val_coco_api = convert_to_coco_api(dataloader_val.dataset, bbox_fmt="yolo")

# In datasets that inherit torchvision.CocoDetection a COCO object is created so we do not have to create one;
# this coco object stores the raw ground truth labels such as bboxes in coco format and original image height/width;
# this is useful because the CocoEvaluator wants the original image dimensions and bboxes in coco format;
# the images can still be resized for validation, however, the final evaluation score should be resized to the original image height
# val_coco_api = dataloader_val.dataset.coco
# coco_evaluator = CocoEvaluator(
#     val_coco_api, iou_types=["bbox"], bbox_format="coco"
# )

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
# for stat in top_stats[:10]:
#     print(stat)

# final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
#    bbox_predictions, targets
# )

# results = val_preds_to_img_size(targets, bbox_preds, class_conf)
evaluator_time = time.time()

# results is a dict containing:
#   "img_id": {boxes: [], "scores": [], "labels", []}
# where scores is the maximum probability for the class (class probs are mulitplied by objectness probs in an earlier step)
# and labels is the index of the maximum class probability; reminder
coco_evaluator.update(results)
evaluator_time = time.time() - evaluator_time

if (steps + 1) % self.log_intervals["train_steps_freq"] == 0:
    log.info(
        "val steps:%d/%-10d ",
        steps + 1,
        len(dataloader_val),
    )
    log.info("cpu utilization: %s", psutil.virtual_memory().percent)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # for stat in top_stats[:10]:
    #     print(stat)

coco_evaluator.synchronize_between_processes()

# Accumulate predictions from all processes
coco_evaluator.accumulate()
coco_evaluator.summarize()

return coco_evaluator
