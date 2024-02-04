import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from detectors.utils import utils
from detectors.vocab import Vocab


class Trainer:
    """Trainer TODO: comment"""

    def __init__(self, output_path: str):
        """Constructor for the Trainer class

        Args:
            corpus: body of text to encode/decode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

        ## TODO: PROBALBY REMOVE THESE Initialize training objects
        # self.optimizer = optimizer_map[optimizer]
        # self.lr_scheduler = "test"

        # Paths
        self.output_paths = {
            "output_dir": Path(
                f"{output_path}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
            ),
        }

    def _train_one_epoch(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        
        for steps, (samples, targets) in enumerate(tqdm(data_loader, ascii=' >=')):
           
            samples = samples.to(device)
            targets = [{key: value.to(device) for key, value in t.items()} for t in targets]
            
            bbox_predictions = model(samples)
            
            ## TOOD: understand this and rename variables if needed
            loss, loss_xy, loss_wh, loss_obj, loss_cls, lossl2 = criterion(bbox_predictions, targets["bboxes"])
            
            
            exit()

    # def train():

    def train(
        self,
        model,
        criterion,
        data_loader,
        optimizer,
        scheduler,
        start_epoch=0,
        epochs=100,
        ckpt_every=None,
        device="cpu",
    ):
        """Train a model

        Args:
            model:
            optimizer:
            ckpt_every:
        """
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            train_stats = self._train_one_epoch(
                model, criterion, data_loader, optimizer, device
            )
            scheduler.step()

            # Save the model every ckpt_every
            if ckpt_every is not None and (epoch + 1) % ckpt_every == 0:
                ckpt_path = self.output_paths["output_dir"] / f"checkpoint{epoch:04}"
                self._save_model(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_every,
                    save_path=ckpt_path,
                )

    def _save_model(
        self, model, optimizer, lr_scheduler, current_epoch, ckpt_every, save_path
    ):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": current_epoch,
            },
            save_path,
        )

    @torch.no_grad()
    def estimate_loss(
        self, train_data, val_data, model, eval_iters, batch_size, block_size
    ) -> Dict[str, float]:
        """Estimate the loss of the train and val split

        Args:

        """
        out = {}
        model.eval()
        all_data = {"train": train_data, "val": val_data}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = Vocab.get_batch(
                    all_data[split], batch_size, block_size, self.device
                )
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
