"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from wandb import Table
import wandb
from PIL import Image
from models.models import (
    ResNet50Pretrained21k,
    ViTPretrained21k,
    MLPMixerPretrained21k,
    BEiTPretrained21k,
)
import numpy as np
import requests
import copy

from torchvision import transforms

timm_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)


class DollarstreetDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        transform=timm_transform,
        return_type="image",
        indexing_type="standard",
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.return_type = return_type
        self.indexing_type = indexing_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        url = row["url"]
        label = row["label"]

        if self.indexing_type == "alibaba":
            label_indices = row["sin21k_targets_alibaba"]
        elif self.indexing_type == "standard":
            label_indices = row["sin21k_targets"]

        if self.return_type == "image":
            image = Image.open(requests.get(row["url"], stream=True).raw)
            if self.transform:
                image = self.transform(image)
            return image, label, {"url": url, "label_indices": label_indices}

        elif self.return_type == "url":
            return str(url), label, {"url": url, "label_indices": label_indices}


class DollarStreetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers=0,
        image_size=224,
        transform=None,
        return_type="image",
        indexing_type="alibaba",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.return_type = return_type
        self.indexing_type = indexing_type

    def test_dataloader(self) -> DataLoader:
        data_loader = self._create_dataloader(stage="test")
        return data_loader

    def _create_dataloader(self, stage: str):
        shuffle = True if stage == "train" else False
        dataset = DollarstreetDataset(
            annotations_file="metadata.csv",
            transform=self.transform,
            return_type=self.return_type,
            indexing_type=self.indexing_type,
        )

        self.dataset = dataset
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader


def make_saving_dir(directory: str, experiment_name: str):
    save_dir = f"{directory}/{experiment_name}"
    if os.path.exists(save_dir):
        print(
            f"WARNING - this experiment folder ({save_dir}) already exists and will be overwritten!"
        )
    else:
        os.mkdir(save_dir)
    return save_dir


def run_eval(
    models: list,
    datamodule: pl.LightningDataModule,
    directory: str = "",
    experiment_name: str = "test",
):
    # Set up logging
    save_dir = make_saving_dir(directory=directory, experiment_name=experiment_name)
    print(f"Saving Directory: {save_dir}")

    for i in range(len(models)):
        model = models[i]
        model_name = model.__class__.__name__
        print(f"\n################ Evaluating {model_name}... ################")
        trainer = pl.Trainer(accelerator="gpu", devices=1)

        if "BEiT" in model_name:
            # Beit takes in a PIL image, not a tensor. Which means we need to make the following changes:
            dm = copy.copy(datamodule)
            dm.return_type = "url"  #  1) return the URL from the dataloader,
            dm.batch_size = 1  # 2) we can only download 1 image at once
            dm.indexing_type = (
                "standard"  # 3) return 22k indices instead of 11k indices
            )

            trainer.test(model=model, datamodule=dm)
        else:
            trainer.test(model=model, datamodule=datamodule)

    return


from models.models import (
    ViTPretrained21k,
    ResNet50Pretrained21k,
    PredictionLogger,
    ViTPretrained21k_MIIL,
    BEiTPretrained21k,
    SeerPretrained,
)
from eval_utils import (
    calculate_top_preds_11k,
    calculate_top_preds_22k,
    calculate_top_preds_22k_no_subset,
)

dm = DollarStreetDataModule(batch_size=32, num_workers=80, transform=timm_transform)

experiment_name = "test"
directory = ""


def make_logger(conversion_type="11k", subset=True):
    if conversion_type == "22k":
        if subset:
            return PredictionLogger(
                top_n_to_store=5,
                write_itermediate_results=False,
                save_dir=f"{directory}/{experiment_name}",
                calculate_top_preds=calculate_top_preds_22k,
            )
        else:
            return PredictionLogger(
                top_n_to_store=5,
                write_itermediate_results=False,
                save_dir=f"{directory}/{experiment_name}",
                calculate_top_preds=calculate_top_preds_22k_no_subset,
            )
    if conversion_type == "11k":
        return PredictionLogger(
            top_n_to_store=5,
            write_itermediate_results=False,
            save_dir=f"{directory}/{experiment_name}",
            calculate_top_preds=calculate_top_preds_11k,
        )


if __name__ == "__main__":

    # Make models
    model1 = ViTPretrained21k(prediction_logger=make_logger())
    model2 = MLPMixerPretrained21k(prediction_logger=make_logger())
    model3 = ResNet50Pretrained21k(prediction_logger=make_logger())
    model4 = ViTPretrained21k_MIIL(prediction_logger=make_logger())
    model5 = BEiTPretrained21k(prediction_logger=make_logger("22k"))
    models = [model1, model2, model3, model4, model5]

    run_eval(models=models, datamodule=dm, experiment_name=experiment_name)
