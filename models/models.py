"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytorch_lightning as pl
import timm
import torch
from transformers import BeitForImageClassification
import torch.nn.functional as F
import pandas as pd
from evaluation.eval_utils import calculate_accuracies
from transformers import BeitForImageClassification, BeitFeatureExtractor
from PIL import Image
import requests
from models.seer_model import BlockParams, partial, RegNet
from transformers import model_zoo
import imagenet1k_to_dollarstreet
import torchvision
from torch import Tensor
from typing import List
import json


class PredictionLogger:
    """Logger for model predictions"""

    def __init__(
        self,
        top_n_to_store: int = 5,
        write_itermediate_results: bool = False,
        save_dir: str = "",
        calculate_top_preds: callable = None,
    ) -> None:
        super().__init__()
        self.top_n_to_store = top_n_to_store
        self.write_intermittent_results = write_itermediate_results
        self.save_dir = save_dir
        self.calculate_top_preds = calculate_top_preds

        self.results = pd.DataFrame({})

    def combine_tensor_list_into_str(self, x):
        l = []

        for i in range(len(x)):
            l.append("|".join(str(e) for e in list(x[i])))

        return l

    def store_preds(
        self,
        ids: list,
        output: torch.Tensor,
        label_indices: list,
        labels: list,
    ):

        top_indices, top_confidences, top_prediction_classes = self.calculate_top_preds(
            output
        )

        # Combine predictions and confidences into strings
        combined_top_prediction_classes = self.combine_tensor_list_into_str(
            top_prediction_classes
        )

        combined_top_indices = self.combine_tensor_list_into_str(
            top_indices  # .detach().cpu().numpy()
        )

        combined_top_confidences = self.combine_tensor_list_into_str(
            top_confidences  # .detach().cpu().numpy()
        )
        output = self.combine_tensor_list_into_str(output.detach().cpu().numpy())

        # Save to results dictionary
        batch_results = pd.DataFrame(
            {
                "ID": ids,
                "Model_Output": output,
                "Prediction_Indices": combined_top_indices,
                "Prediction_Classes": combined_top_prediction_classes,
                "Prediction_Confidences": combined_top_confidences,
                "Label_Classes": labels,
                "Label_Indices": label_indices,
            }
        )

        self.results = pd.concat([self.results, batch_results])
        if self.write_intermittent_results:
            self.logger.save_prediction_csv()

    def save_prediction_csv(self, prefix: str):
        self.results = calculate_accuracies(self.results)
        save_path = f"{self.save_dir}/{prefix}_results.csv"
        print(f"\nSaving prediction csv to {save_path}")
        self.results.to_csv(save_path, index=False)
        return self.results


class BaseModel(pl.LightningModule):
    """Basic model used to define basic functionality of models used (including evaluation, saving predictions)"""

    def __init__(
        self,
        index_to_class_map: dict = None,
        prediction_logger: PredictionLogger = None,
    ) -> None:
        super().__init__()
        # Prediction
        self.model = self.load_backbone()
        self.feature_dim = 768
        self.num_classes = 11221

        # Saving
        self.index_to_class_map = index_to_class_map
        self.prediction_logger = prediction_logger

    def load_backbone(self):
        vit = timm.create_model("vit_base_patch16_224_miil_in21k", pretrained=True)
        return vit

    def forward(self, x):
        out = self.model(x)
        return out

    def test_step(self, batch, batch_idx):
        x, y, metadata = batch
        y_hat = self.forward(x)

        if self.prediction_logger:
            self.prediction_logger.store_preds(
                ids=metadata["url"],
                label_indices=metadata["label_indices"],
                output=y_hat,
                labels=y,
            )
        else:
            return

    def test_epoch_end(self, outputs):
        if self.prediction_logger:
            self.prediction_logger.save_prediction_csv(prefix=self.__class__.__name__)


class ViTPretrained21k(BaseModel):
    """ViT with Aug Reg pretrained on ImageNet 21k"""

    def load_backbone(self):
        # trained on ImageNet-21k with aug reg
        # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L126
        vit = timm.create_model("vit_base_patch16_224_miil_in21k", pretrained=True)
        return vit


class MLPMixerPretrained21k(BaseModel):
    def load_backbone(self):
        model = timm.create_model("mixer_b16_224_miil_in21k", pretrained=True)
        return model


class ResNet50Pretrained21k(BaseModel):
    def load_backbone(self):
        checkpoint_url = "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth"
        model = timm.create_model("resnet50", pretrained=False, num_classes=11221)
        state_dict = torch.utils.model_zoo.load_url(checkpoint_url)["state_dict"]
        model.load_state_dict(state_dict)
        return model

    def rla(self):
        print("rla found")


class ViTPretrained21k_MIIL(BaseModel):
    def load_backbone(self):
        checkpoint_url = "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth"
        model = timm.create_model(
            "vit_base_patch16_224_miil_in21k", pretrained=False, num_classes=11221
        )
        state_dict = torch.utils.model_zoo.load_url(checkpoint_url)["state_dict"]
        model.load_state_dict(state_dict)
        return model


class BEiTPretrained21k(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # note the number of classes is different from the others
        self.num_classes = 21841
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(
            "microsoft/beit-large-patch16-224"
        )

    def load_backbone(self):

        model = BeitForImageClassification.from_pretrained(
            "microsoft/beit-large-patch16-224-pt22k-ft22k"
        )

        return model

    def forward(self, x):
        url_str = x[0]
        im = Image.open(requests.get(url_str, stream=True).raw)
        inputs = self.feature_extractor(images=im, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs).logits

        return outputs


class ResNet50Pretrained1k(ResNet50Pretrained21k):
    def __init__(self, model_name_suffix: str = "", save_dir: str = "data/texture/"):
        super().__init__()
        self.num_classes = 1000
        self.model_name_suffix = model_name_suffix
        self.save_dir = save_dir

        # task_media_id -> {"top5_imagenet1k_indices": [], "top5_imagenet1k_probs": []}
        self.task_id_to_predictions = {}

        valid_imagenet1k_indices_set = (
            imagenet1k_to_dollarstreet.ValidImageNet1kIndices().get_valid_indices()
        )

        self.valid_imagenet1k_indices = list(valid_imagenet1k_indices_set).sort()

    def load_backbone(self):
        model = torchvision.models.resnet50(pretrained=True)
        return model

    def test_step(self, batch, batch_idx):
        x, y, info = batch
        out = self.model(x)[:, self.valid_imagenet1k_indices]
        y_hat = F.softmax(out)

        top5_results = torch.topk(y_hat, 5)
        top5_indices, top5_probs = top5_results.indices, top5_results.values
        self.store_top_predictions(info["task_media_id"], top5_indices, top5_probs)
        return

    def on_test_end(self):
        self.save_predictions_to_json()

    def store_top_predictions(
        self, task_ids: List[str], pred_indices: Tensor, pred_probs: Tensor
    ) -> dict:
        task_id_to_predictions = {}
        for i, task_id in enumerate(task_ids):
            pred_idx = pred_indices[i, :].tolist()
            pred_prob = pred_probs[i, :].tolist()
            results = {"pred_indices": pred_idx, "pred_prob": pred_prob}
            task_id_to_predictions[task_id] = results

        self.task_id_to_predictions.update(task_id_to_predictions)
        return task_id_to_predictions

    def save_predictions_to_json(self) -> None:
        file_path = (
            f"{self.save_dir}{self.__class__.__name__}_{self.model_name_suffix}.json"
        )
        with open(file_path, "w") as fp:
            json.dump(self.task_id_to_predictions, fp)


class ResNet50TextureDebiased1k(ResNet50Pretrained1k):
    def __init__(self, model_name_suffix: str = ""):
        super().__init__(model_name_suffix=model_name_suffix)

    def load_backbone(self):
        """Loading ResNet without texture bias based on https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py"""
        model_url = "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar"
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_url)
        model.load_state_dict(checkpoint["state_dict"])
        return model


if __name__ == "__main__":
    vit = ViTPretrained21k()
    x = torch.rand(8, 3, 224, 224)
    print("vit output shape", vit(x).shape)

    mlp_mixer = MLPMixerPretrained21k()
    print("mlp mixer output shape", mlp_mixer(x).shape)

    resnet = ResNet50Pretrained21k()
    print("resnet50 output shape", resnet(x).shape)

    beit = BEiTPretrained21k()
    print("BEiT output shape", beit(x).shape)
