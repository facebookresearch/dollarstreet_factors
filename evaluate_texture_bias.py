"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Compares a ResNet50  with or without a texture bias.

Stores json of top 5 predictions in data/texture/[model].json

"""

from torchvision import transforms
from PIL import Image
import os
import requests
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from annotations import Annotations
from typing import List
import torch
import imagenet1k_to_dollarstreet
from models.models import ResNet50Pretrained1k, ResNet50TextureDebiased1k
import pytorch_lightning as pl


class DollarstreetDataset(Dataset):

    timm_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )

    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        annotations: Annotations,
        data_dir: str = "data_dir/",
        transform=timm_transform,
        return_type="image",
        indexing_type="standard",
    ):
        self.annotations = annotations
        self.data_dir = data_dir
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = annotations.annotations
        self.transform = transform
        self.return_type = return_type
        self.indexing_type = indexing_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        url = row["url"]
        label = row["class"]
        task_media_id = row.name

        image_name = url.split("/")[-1]
        image_path = os.path.join(self.data_dir, image_name)

        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.open(requests.get(url, stream=True).raw)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(0),
            {
                "url": url,
                "task_media_id": task_media_id,
                "label": label,
            },
        )


def generate_predictions():
    """Saves predictions to json for each model"""
    ds = DollarstreetDataset(
        Annotations(),
        # transform=DollarstreetDataset.imagenet_transform
    )
    dl = DataLoader(ds, 32, shuffle=False)

    model = ResNet50Pretrained1k(model_name_suffix="imagenet_transforms")
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, dl)

    model = ResNet50TextureDebiased1k(model_name_suffix="imagenet_transforms")
    trainer.test(model, dl)


class Analysis:
    """Compares the performance of ResNet to ResNet texture debiased.

    1. Filters DollarStreet indices with valid ImageNet-1k mapping
    2. Compute top5 accuracy for each image (if any top-5 prediction is correct)
    """

    def __init__(
        self,
        results_dir: str = "data/texture/",
        model_name_suffix: str = "",
    ):
        self.results_dir = results_dir
        self.model_name_suffix = model_name_suffix

        self.annotations = Annotations()
        table = self.annotations.annotations
        self.table = self.filter_images_with_valid_1k_mapping(table)
        self.table_multilabel = self.convert_to_multilabel(self.table)

        self.resnet_predictions: dict = self.load_resnet_predictions(
            f"ResNet50Pretrained1k{model_name_suffix}"
        )
        self.resnet_texture_debiased_predictions: dict = self.load_resnet_predictions(
            f"ResNet50TextureDebiased1k{model_name_suffix}"
        )

        self.add_predictions_to_multilabel_table(self.resnet_predictions, "resnet")
        self.add_predictions_to_multilabel_table(
            self.resnet_texture_debiased_predictions, "resnet_texture_debiased"
        )

    def load_resnet_predictions(self, model_name: str = "ResNet50Pretrained1k") -> dict:
        file_path = os.path.join(self.results_dir, f"{model_name}.json")
        with open(file_path, "r") as read_file:
            predictions = json.load(read_file)
        return predictions

    def filter_images_with_valid_1k_mapping(self, table: pd.DataFrame) -> pd.DataFrame:
        """Filter leaves 6705 from 13k images"""
        valid_dollarstreet_classes = set()
        for (
            dollarstreet_name,
            imagenet1k_names,
        ) in imagenet1k_to_dollarstreet.DOLLARSTREET_TO_IMAGENET1K_NAME.items():
            if imagenet1k_names:
                valid_dollarstreet_classes.add(dollarstreet_name)

        table = table[table["class"].isin(valid_dollarstreet_classes)]
        return table

    def convert_to_multilabel(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()

        url_to_classes = table.groupby("url")["class"].apply(list)

        table = table.drop_duplicates(subset=["url"])
        table = table.drop("class", axis=1)

        return table.join(url_to_classes, on="url")

    def add_predictions_to_multilabel_table(
        self, predictions: dict, name: str
    ) -> pd.DataFrame:
        """Updates multilabel table with predictions"""

        table = self.table_multilabel

        for task_id, dollarstreet_classes, url in zip(
            table.index, table["class"], table["url"]
        ):
            table.loc[task_id, f"is_correct_{name}"] = False
            table.loc[task_id, f"pred_prob_{name}"] = 0

            pred_indices = predictions[task_id]["pred_indices"][0]
            pred_probs = predictions[task_id]["pred_prob"][0]
            pred_names = [
                imagenet1k_to_dollarstreet.IMAGENET1K_IDX_TO_NAMES[idx]
                for idx in pred_indices
            ]

            true_imagenet_names = set()
            for dollarstreet_class in dollarstreet_classes:
                imagenet_names = (
                    imagenet1k_to_dollarstreet.DOLLARSTREET_TO_IMAGENET1K_NAME[
                        dollarstreet_class.replace("_", " ")
                    ]
                )
                true_imagenet_names.update(imagenet_names)

            for pred_name, pred_prob in zip(pred_names, pred_probs):
                if pred_name in true_imagenet_names:
                    table.loc[task_id, f"is_correct_{name}"] = True
                    table.loc[task_id, f"pred_prob_{name}"] = pred_prob
        return table

    def print_overall_accuracies(
        self,
    ):
        print("ResNet overall", self.table_multilabel["is_correct_resnet"].mean())
        print(
            "ResNet texture debiased overall",
            self.table_multilabel["is_correct_resnet_texture_debiased"].mean(),
        )
        print()

        print("For images marked with texture as distinctive factor")

        table = self.table_multilabel[self.table_multilabel["texture"] == 1]

        print(
            "ResNet",
            table.groupby("income_bucket")["is_correct_resnet"].mean(),
        )
        print(
            "ResNet texture debiased",
            table.groupby("income_bucket")["is_correct_resnet_texture_debiased"].mean(),
        )

        print(
            "ResNet",
            table.groupby("region")["is_correct_resnet"].mean(),
        )

        print(
            "ResNet texture debiased",
            table.groupby("region")["is_correct_resnet_texture_debiased"].mean(),
        )

        print(
            "Number of images marked with texture per region",
            table.groupby("region")["can_i_rate_this_job"].count(),
        )

        print(
            "Number of images marked with texture per income bucket",
            table.groupby("income_bucket")["can_i_rate_this_job"].count(),
        )


if __name__ == "__main__":
    # generate_predictions()

    # analysis = Analysis(model_name_suffix="_imagenet_transforms")
    analysis = Analysis()
    analysis.print_overall_accuracies()
