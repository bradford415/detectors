import datetime
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import data

from detectors.evaluate import evaluate, evaluate_detr, load_model_checkpoint
from detectors.utils import distributed
from detectors.visualize import plot_loss, plot_mAP, visualize_norm_img_tensors

log = logging.getLogger(__name__)


# TODO: consider making this a base class and creating different trainer objects for yolo/detr
class Trainer:
    """Trainer TODO: comment"""

    def __init__(
        self,
        output_dir: str,
        model_name: str,
        step_lr_on: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        log_train_steps: int = 20,
    ):
        """Constructor for the Trainer class

        Args:
            output_path: Path to save the train outputs
            model_name: the name of the model being trained; this determines which logic to use
            step_lr_on: whether to call scheduler.step every 'step' or 'epoch'
            use_cuda: Whether to use the GPU
        """
        self.device = device

        self.output_dir = Path(output_dir)
        self.log_train_steps = log_train_steps

        self.step_lr_on = step_lr_on
        self.enable_amp = True if not self.device.type == "mps" else False
        self.model_name = model_name

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: data.DataLoader,
        dataloader_val: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        class_names: list[str],
        grad_accum_steps: int,
        coco_api: Optional[COCO] = None,
        postprocessors: Optional[dict] = None,
        max_norm: Optional[float] = None,
        start_epoch: int = 1,
        epochs: int = 100,
        ckpt_epochs: int = 10,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Trains a model

        Specifically, this method trains a model for n epochs and evaluates on the validation set.
        A model checkpoint is saved at user-specified intervals

        Args:
            model: A pytorch model to be trained
            criterion: The loss function to use for training
            dataloader_train: Torch dataloader to loop through the train dataset
            dataloader_val: Torch dataloader to loop through the val dataset
            optimizer: Optimizer which determines how to update the weights
            class_names: list of class names; used for logging and visualization
            grad_accum_steps: number of steps to accumulate gradients before updating the weights;
                              used to simulate a larger effective batch size
            coco_api: TODO
            postprocessors: postprocessing that needs to be applied after validation/inference;
                            e.g., convert a models normalized outputs to the original image size
            max_norm: the value to clip the norm of gradients if magnitude of the gradients
                      is above max_norm; ((grad) / ||grad||) * max_norm
            scheduler: Scheduler which determines how to change the learning rate
            start_epoch: Epoch to start the training on; starting at 1 is a good default because it makes
                         checkpointing and calculations more intuitive
            epochs: The epoch to end training on; unless starting from a check point, this will be the number of epochs to train for
            ckpt_every: Save the model after n epochs
        """
        log.info("\ntraining started\n")

        if checkpoint_path is not None:
            start_epoch = load_model_checkpoint(
                checkpoint_path, model, optimizer, self.device, scheduler
            )
            log.info(
                "NOTE: A checkpoint file was provided, the model will resume training at epoch %d",
                start_epoch,
            )

        total_train_start_time = time.time()

        last_best_path = None

        scaler = torch.amp.GradScaler(self.device.type)

        best_ap = 0.0
        train_loss = []
        val_loss = []
        epoch_mAP = []
        for epoch in range(start_epoch, epochs + 1):
            model.train()

            # Track the time it takes for one epoch (train and val)
            one_epoch_start_time = time.time()

            # Train one epoch
            if "yolo" in self.model_name:
                epoch_train_loss = self._train_one_epoch_yolo(
                    model,
                    criterion,
                    dataloader_train,
                    optimizer,
                    scheduler,
                    epoch,
                    grad_accum_steps,
                    scaler,
                )
                train_loss.append(epoch_train_loss)
            else:
                # train one epoch of a detr-based model; returns a dict of loss components
                epoch_train_loss = self._train_one_epoch_detr(
                    model,
                    criterion,
                    dataloader_train,
                    optimizer,
                    scheduler,
                    epoch,
                    grad_accum_steps,
                    max_norm,
                    scaler,
                )

            curr_lr = optimizer.param_groups[0]["lr"]

            # Increment lr scheduler every epoch if set for "epochs"
            if scheduler is not None and self.step_lr_on == "epochs":
                scheduler.step()

            # Save a checkpoint before the LR drop; this is beneficial for several reasons:
            #   1. Fallback point: if training after the LR drop doesn't improve performance or
            #      overfits, you can go back to this checkpoint and try different strategies
            if epoch % scheduler.step_size == 0:
                ckpt_path = (
                    self.output_dir
                    / "checkpoints"
                    / f"checkpoint{epoch:04}_lr_{str(curr_lr).replace('.', '-')}.pt"
                )
                self._save_model_master(
                    model, optimizer, epoch, save_path=ckpt_path, lr_scheduler=scheduler
                )

            # Save the model every ckpt_epochs
            if epoch % ckpt_epochs == 0:
                ckpt_path = self.output_dir / "checkpoints" / f"checkpoint{epoch:04}.pt"
                self._save_model_master(
                    model, optimizer, epoch, save_path=ckpt_path, lr_scheduler=scheduler
                )

            # Evaluate the model on the validation set
            log.info("\nEvaluating on validation set — epoch %d", epoch)

        # TODO: probably save metrics output into csv
        breakpoint()
        if self.model_name == "dino":
            stats = evaluate_detr(
                model,
                dataloader_val,
                coco_api,
                class_names,
                postprocessors,
                criterion=criterion,
                enable_amp=self.enable_amp,
                output_path=self.output_dir,
                device=self.device,
            )
        else:
            # evaluate() is used by both val and test set; this can be customized in the future if needed
            # but for now validation and test behave the same
            metrics_output, detections, val_loss = evaluate(
                model,
                dataloader_val,
                class_names,
                criterion=criterion,
                output_path=self.output_dir,
                device=self.device,
            )

            ## START HERE
            val_loss.append(epoch_val_loss.item())

            precision, recall, AP, f1, ap_class = metrics_output
            mAP = AP.mean()
            epoch_mAP.append(mAP * 100)

            plot_loss(train_loss, val_loss, save_dir=str(self.output_dir))
            plot_mAP(epoch_mAP, save_dir=str(self.output_dir))

            # Create csv file of training stats per epoch
            train_dict = {
                "epoch": list(np.arange(start_epoch, epoch + 1)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mAP": epoch_mAP,
            }
            pd.DataFrame(train_dict).to_csv(
                self.output_dir / "train_stats.csv", index=False
            )

            # Save and overwrite the checkpoint with the highest val mAP
            if round(mAP, 4) > round(best_ap, 4):
                best_ap = mAP

                mAP_str = f"{mAP*100:.2f}".replace(".", "-")
                best_path = self.output_dir / "checkpoints" / f"best_mAP_{mAP_str}.pt"

                log.info(
                    "new best mAP of %.2f found at epoch %d; saving checkpoint",
                    mAP * 100,
                    epoch,
                )
                self._save_model_master(
                    model, optimizer, epoch, save_path=best_path, lr_scheduler=scheduler
                )

                # delete the previous best mAP model's checkpoint
                if last_best_path is not None:
                    last_best_path.unlink(missing_ok=True)
                last_best_path = best_path

            # Uncomment to visualize validation detections
            # save_dir =  / "validation" / f"epoch{epoch}"
            # plot_all_detections(image_detections, classes=class_names, output_dir=save_dir)

            # Current epoch time (train/val)
            one_epoch_time = time.time() - one_epoch_start_time
            one_epoch_time_str = str(datetime.timedelta(seconds=int(one_epoch_time)))
            log.info("\nEpoch time (h:mm:ss): %s", one_epoch_time_str)

        # Entire training time
        total_time = time.time() - total_train_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info(
            "Training time for %d epochs (h:mm:ss): %s ",
            start_epoch - epochs,
            total_time_str,
        )

    def _train_one_epoch_yolo(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        grad_accum_steps: int,
        scaler: torch.amp,
    ):
        """Train one epoch

        Args:
            model: Model to train
            criterion: Loss function
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
            scheduler: Learning rate scheduler to update the learning rate
            epoch: Current epoch; used for logging purposes
            grad_accum_steps: number of steps to accumulate gradients before updating the weights;
                              the loss will be divivided by this number to account for the accumulation
        """
        epoch_loss = []
        for steps, (samples, targets) in enumerate(dataloader_train, 1):
            samples = samples.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.enable_amp,
            ):
                # list of preds at all 3 scales;
                # bbox_preds[i] (B, (5+n_class)*num_anchors, out_w, out_h)
                bbox_preds = model(samples)

                # final_loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(
                #     bbox_preds, targets, model
                # ) # yolov4
                # loss_components = misc.to_cpu(
                #     torch.stack([loss_xy, loss_wh, loss_obj, loss_cls, lossl2])
                # )

                total_loss, loss_components = criterion(bbox_preds, targets, model)

                if grad_accum_steps > 1:
                    # scale the loss by the number of accumulation steps
                    loss = loss / grad_accum_steps

                # multiply loss by the mini batch size to account for split gradients; I'm not entirely sure how this works
                # but it was mentioned here: https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/818#issuecomment-1484223518
                # I don't think this is correct;
                # i think this is only used if you want to sum the loss instead of average bc this cancels out the average
                total_loss *= samples.shape[0]

            # Calculate gradients
            if self.enable_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            epoch_loss.append(total_loss.detach().cpu())

            # Update the gradients once all the subdivisions have finished accumulating gradients and update lr_scheduler
            # TODO: verify this is accurate
            if steps % grad_accum_steps == 0:
                # Calculate gradients and updates weights
                if self.enable_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                # NOTE: occasionally scaler.step will be skipped and torch will throw a warning that scheduler.step
                #       is being called first; I couldn't find a great solution but I think it's safe to ignore
                if scheduler is not None:
                    scheduler.step()

            # Calling scheduler step increments a counter which is passed to the lambda function;
            # if .step() is called after every batch, then it will pass the current step;
            # if .step() is called after every epoch, then it will pass the epoch number;
            # this counter is persistent so every epoch it will continue where it left off i.e., it will not reset to 0
            if (steps) % 100 == 0:
                log.info(
                    "Current learning_rate: %s\n",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                )

            if (steps) % self.log_train_steps == 0:
                log.info(
                    "epoch: %-10d iter: %d/%-12d train_loss: %-10.4f bbox_loss: %-10.4f obj_loss: %-10.4f class_loss: %-10.4f",
                    epoch,
                    steps,
                    len(dataloader_train),
                    loss_components[3],
                    loss_components[0],
                    loss_components[1],
                    loss_components[2],
                )

        # TODO: see if this is correct
        return np.array(epoch_loss).mean() / samples.shape[0] / grad_accum_steps

    def _train_one_epoch_detr(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader_train: Iterable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epoch: int,
        grad_accum_steps: int,
        max_norm: float,
        scaler: torch.amp,
    ):
        """Train one epoch

        Args:
            model: Model to train
            criterion: Loss function
            dataloader_train: Dataloader for the training set
            optimizer: Optimizer to update the models weights
            scheduler: Learning rate scheduler to update the learning rate
            epoch: Current epoch; used for logging purposes
            grad_accum_steps: number of steps to accumulate gradients before updating the weights;
                              the loss will be divivided by this number to account for the accumulation
            max_norm: the value to clip the norm of gradients if magnitude of the gradients
                      is above max_norm; ((grad) / ||grad||) * max_norm

        Returns:
            a dictionary of the averaged, scaled, loss components which is the loss average for the
            entire batch; not every loss component is used in the gradient computation; the
            "loss" key is the total loss used for backpropagation, averaged across the epoch
        """
        epoch_loss = []
        running_loss_dict = {}  # TODO should make these default dicts
        epoch_loss_dict = {}
        num_update_steps = 0
        for steps, (samples, targets) in enumerate(dataloader_train, 1):
            samples = samples.to(self.device)

            # move label tensors to gpu
            targets = [
                {
                    key: (val.to(self.device) if isinstance(val, torch.Tensor) else val)
                    for key, val in t.items()
                }
                for t in targets
            ]

            accumulate_grads = not steps % grad_accum_steps == 0

            with self._maybe_no_sync(model, sync=not accumulate_grads):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.enable_amp,
                ):
                    preds = model(samples, targets)

                    loss_dict = criterion(preds, targets)
                    weight_dict = criterion.weight_dict

                    # compute the total loss by scaling each component of the loss by its weight value;
                    # if the loss key is not a key in the weight_dict, then it is not used in the total loss;
                    # dino sums a total of 39 losses w/ the default values;
                    # see detectors/models/README.md for information on the losses that propagate gradients
                    loss = sum(
                        loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys()
                        if k in weight_dict
                    )

                    if grad_accum_steps > 1:
                        # scale the loss by the number of accumulation steps
                        loss = loss / grad_accum_steps
                        # TODO I think i need to sum the dicts here before reducing for logging

                    # summing the averaged loss components for gradient accumulation so we don't need to
                    # account for this at the end; resets every update step
                    for key, val in loss_dict.items():
                        if key in running_loss_dict:
                            running_loss_dict[key] += val.detach() / grad_accum_steps
                        else:
                            running_loss_dict[key] = val.detach() / grad_accum_steps

                # Calculate gradients; NOTE: DDP averages the gradients across proccesses such that every
                # process has the same gradients and updates the weights the same
                # (does an element-wise sum on the gradients and divides by world_size to average)
                if self.enable_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            ## Remove this maybe? ###
            epoch_loss.append(loss.detach().cpu())

            # Update the gradients once finished accumulating gradients and update lr_scheduler
            if not accumulate_grads:
                num_update_steps += 1

                # average the losses across all processes; represnets the current step loss
                # (sums the accumulated gradients)
                reduced_loss_dict = distributed.reduce_dict(
                    running_loss_dict, average=True
                )

                # scale the loss components just like the total loss computation; this is
                # the average loss across all processes
                reduced_loss_dict_scaled = {
                    k: v * weight_dict[k]
                    for k, v in reduced_loss_dict.items()
                    if k in weight_dict
                }

                for key, val in reduced_loss_dict_scaled.items():
                    if key in epoch_loss_dict:
                        epoch_loss_dict[key] += val.detach()
                    else:
                        epoch_loss_dict[key] = val.detach()

                # NOTE: we use this total loss instead of the one use for backpropagation
                #       because this one is an average across processes
                average_loss_scaled = sum(reduced_loss_dict_scaled.values()).item()

                # clip gradients if needed and update weights
                if self.enable_amp:
                    if max_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()

                optimizer.zero_grad()

                # reset dict for next step
                running_loss_dict = {}

            if (steps) % 100 == 0:
                # TODO: might need to update this since we use two param groups
                log.info(
                    "Current learning_rate: %s\n",
                    optimizer.state_dict()["param_groups"][0]["lr"],
                )

            if (steps) % self.log_train_steps == 0:
                log.info(
                    "epoch: %-10d iter: %d/%-12d train_loss: %-10.4f",
                    epoch,
                    steps,
                    len(dataloader_train),
                    average_loss_scaled,
                )

        avg_epoch_loss = {k: v / num_update_steps for k, v in epoch_loss_dict.items()}
        avg_epoch_loss["loss"] = average_loss_scaled
        # TODO: see if this is correct

        return avg_epoch_loss

    def _maybe_no_sync(self, model: nn.Module, sync: bool = True):
        """Decides whether to enable no_sync() which avoids synchronizing the gradients accross processes.

        When performing gradient accumulation while using DDP, if we do not disable gradient synching
        then every step we accumulate gradients there will be a lot of communication overhead. A better
        approach would be to only sync gradients on the step when we update our model's weights,
        when we finish accumulating gradients. Additionally, if we do not want to use DDP, e.g., if we
        only have 1 gpu, then no_sync will not be recognized so we can pass nullcontext() instead to
        skip it.

        Addtional reading on no_synch and gradient accumulation:
            https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization

        Args:
            model: the model being trained
            sync: whether to synchronize gradients
        """
        if dist.is_initialized() and not sync:
            # if using DDP and accumulating gradients
            return model.no_sync()
        else:
            # if not using DDP or using DDP and synching gradients as normal
            return nullcontext()

    def _save_model_master(
        self,
        model,
        optimizer,
        current_epoch,
        save_path,
        lr_scheduler: Optional[nn.Module] = None,
    ):
        """Save the model on the master process
        TODO flesh this out more

        Args:
            model: the model to save with or without the DDP wrapper
            TODO
        """
        # extract the model without DDP wrapper if it exists; we do not want to save the DDP wrapper
        model_to_save = model.module if hasattr(model, "module") else model

        if distributed.is_main_process():
            save_path.parents[0].mkdir(parents=True, exist_ok=True)

            save_dict = {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": current_epoch
                + 1,  # + 1 bc when we resume training we want to start at the next step
            }
            if lr_scheduler is not None:
                save_dict["lr_scheduler"] = lr_scheduler.state_dict()

            torch.save(
                save_dict,
                save_path,
            )
