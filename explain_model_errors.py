"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from annotations import Annotations
import os
import pandas as pd

CLIP_MODELS = ["clip_vit_b32", "clip_resnet101", "clip_resnet50_64"]
IN21k_MODELS = [
    "ViTPretrained21k",
    "BEiTPretrained21k",
    "MLPMixerPretrained21k",
    "ResNet50Pretrained21k",
]
MODELS = IN21k_MODELS + CLIP_MODELS


class ModelExplanations:
    """Explains model mistakes using annotations"""

    def __init__(
        self,
        model_name: str = "ViTPretrained21k",
        predictions_csv_dir: str = "predictions_dir/",
        predictions_merge_column: str = "ID",
        annotations_merge_column: str = "url",
    ):
        self.model_name = model_name
        self.predictions_csv_dir = predictions_csv_dir
        self.predictions_merge_column = predictions_merge_column
        self.annotations_merge_column = annotations_merge_column
        if model_name in CLIP_MODELS:
            self.predictions_csv_dir = "data/raw/acc_per_img/"
            self.predictions_merge_column = "url"
        else:
            self.model_name += "_results"

        self.predictions_df = self.load_predictions()
        try:
            self.annotations = Annotations()
        except FileNotFoundError:
            self.annotations = Annotations(data_dir="../data")

        self.factors = self.annotations.factors
        self.annotations_df = self.annotations.annotations

        self.annotated_predictions = self.merge_annotations()

    def load_predictions(self) -> pd.DataFrame:
        file_path = os.path.join(self.predictions_csv_dir, f"{self.model_name}.csv")
        assert os.path.exists(file_path), f"{file_path} not found"
        predictions_df = pd.read_csv(file_path)
        print(file_path)
        return predictions_df[["url", "acc1", "acc5"]]

    def merge_annotations(self) -> pd.DataFrame:
        annotated_predictions = pd.merge(
            how="inner",
            left=self.predictions_df,
            right=self.annotations_df,
            left_on=self.predictions_merge_column,
            right_on=self.annotations_merge_column,
        )
        return annotated_predictions


class CLIPModelExplanations(ModelExplanations):
    """Explains model mistakes using annotations. Defaults to paths/arguments needed for CLIP.
    Currently supported CLIP models are: clip_vit_b32
    """

    def __init__(
        self,
        model_name: str = "clip_vit_b32",
        predictions_csv_dir: str = "data/raw/acc_per_img/",
        predictions_merge_column: str = "url",
        annotations_merge_column: str = "url",
    ):
        super().__init__(
            model_name=model_name,
            predictions_csv_dir=predictions_csv_dir,
            predictions_merge_column=predictions_merge_column,
            annotations_merge_column=annotations_merge_column,
        )


class ErrorRatio:
    def __init__(self, model_explanations: ModelExplanations):
        self.model_explanations = model_explanations

        self.annotated_predictions = model_explanations.annotated_predictions
        self.factors = model_explanations.factors

        self.factor_counts_total: pd.Series = self.annotated_predictions[
            self.factors
        ].sum()

    @property
    def factor_counts_incorrect(self) -> pd.Series:
        """Returns the sum of factor counts for incorrect predictions by top 5"""
        incorrect = self.annotated_predictions[self.annotated_predictions["acc5"] == 0]
        return incorrect[self.factors].sum()

    def _compute_error_ratio(
        self, factor_counts_group: pd.Series, factor_counts_total: pd.Series
    ) -> pd.Series:
        """Computes the error ratio from ImageNetX to measure factors associated with model mistakes.
        Filters factors with fewer than 5 samples

        Args:
            factor_counts_group: series with counts for each factor for the subgroup of interest
            factor_counts_total: series with counts for each factor on the aggregate.

        Returns: a series with likelihood a factor is more associated with mistakes.
        """
        factor_counts_group = self.filter_low_count_factors(factor_counts_group)
        factor_counts_total = self.filter_low_count_factors(factor_counts_total)

        factor_ratio_group = factor_counts_group / factor_counts_group.sum()
        factor_ratio_total = factor_counts_total / factor_counts_total.sum()
        error_ratio = (factor_ratio_group - factor_ratio_total) / factor_ratio_total
        error_ratio = error_ratio.sort_values(ascending=False)
        return error_ratio

    def filter_low_count_factors(self, factor_counts: pd.Series) -> pd.Series:
        factor_counts[factor_counts <= 5] = 0
        return factor_counts

    def compute_error_ratio_total(self) -> pd.Series:
        return self._compute_error_ratio(
            self.factor_counts_incorrect, self.factor_counts_total
        )

    def compute_error_ratio_income_bucket(
        self, income_bucket: str = "low"
    ) -> pd.Series:
        """
        Computes how much more likely factor is associated with mistakes
            for the given income bucket
        Args:
            income_bucket: low, medium, or high
        """
        group = self.annotated_predictions[
            self.annotated_predictions["income_bucket"] == income_bucket
        ]
        factor_counts_in_mistakes = group[group["acc5"] == 0][self.factors].sum()
        return self._compute_error_ratio(
            factor_counts_in_mistakes, self.factor_counts_total
        )

    def compute_error_ratio_region(self, region: str = "Asia") -> pd.Series:
        """
        Computes how much more likely factor is associated with mistakes
            for the given region
        Args:
            region: 'Asia', 'The Americas', 'Europe', 'Africa'
        """
        group = self.annotated_predictions[
            self.annotated_predictions["region"] == region
        ]
        factor_counts_in_mistakes = group[group["acc5"] == 0][self.factors].sum()
        return self._compute_error_ratio(
            factor_counts_in_mistakes, self.factor_counts_total
        )

    def compute_error_ratio_country(self, country: str = "Bangladesh") -> pd.Series:
        """
        Computes how much more likely factor is associated with mistakes
            for the given country
        Args:
            region:
        """
        group = self.annotated_predictions[
            self.annotated_predictions["country"] == country
        ]
        factor_counts_in_mistakes = group[group["acc5"] == 0][self.factors].sum()
        return self._compute_error_ratio(
            factor_counts_in_mistakes, self.factor_counts_total
        )

    def compute_error_ratio_class_region_pair(
        self, class_label: str, region: str
    ) -> pd.Series:
        """
        Computes how much more likely factor is associated with mistakes
            for the given region and class
        Args:
            region: 'Asia', 'The Americas', 'Europe', 'Africa'
            class_label: text label based on DollarStreet class
        """
        mask = (self.annotated_predictions["region"] == region) & (
            self.annotated_predictions["class"] == class_label
        )
        group = self.annotated_predictions[mask]
        factor_counts_in_mistakes = group[group["acc5"] == 0][self.factors].sum()
        return self._compute_error_ratio(
            factor_counts_in_mistakes, self.factor_counts_total
        )

    def compute_error_ratio_class_income_bucket_pair(
        self, class_label: str, income_bucket: str
    ) -> pd.Series:
        """
        Computes how much more likely factor is associated with mistakes
            for the given incombe bucket and class
        Args:
            region: low, medium, high
            class_label: text label based on DollarStreet class
        """
        mask = (self.annotated_predictions["income_bucket"] == income_bucket) & (
            self.annotated_predictions["class"] == class_label
        )
        group = self.annotated_predictions[mask]
        factor_counts_in_mistakes = group[group["acc5"] == 0][self.factors].sum()
        return self._compute_error_ratio(
            factor_counts_in_mistakes, self.factor_counts_total
        )

    def show_top_factor_error_ratio_by_country(self) -> str:
        countries = self.annotated_predictions.country.unique()
        data = []

        for country in countries:
            error_ratios = self.compute_error_ratio_country(country=country)
            top_factor, top_ratio = error_ratios.index[0], error_ratios.values[0]

            entry = {
                "Country": country,
                "Most vulnerable factor": top_factor,
                "Error Ratio": top_ratio,
            }
            data.append(entry)

        df = pd.DataFrame(data)
        df = df.dropna()
        return df.round(2).to_markdown(index=False, tablefmt="pretty")


class CompareFactorAccuracyAcrossModels:
    """Comapres the accuracy on images marked with style factor for ViT versus ResNet CLIP."""

    def __init__(self, data_dir="data/"):
        predictions_csv_dir = os.path.join(data_dir, "raw/acc_per_img")
        self.clip_vit = CLIPModelExplanations(
            model_name="clip_vit_b32", predictions_csv_dir=predictions_csv_dir
        )
        self.clip_resnet = CLIPModelExplanations(
            model_name="clip_resnet101", predictions_csv_dir=predictions_csv_dir
        )

        self.annotated_predictions_vit_clip = self.clip_vit.annotated_predictions
        self.annotated_predictions_resnet_clip = self.clip_resnet.annotated_predictions

    def compute_style_accuracy(
        self, annotated_predictions: pd.DataFrame, factor: str = "larger"
    ) -> float:
        style_images = annotated_predictions[annotated_predictions[factor] == 1]
        correct = len(style_images[style_images["acc5"] == 1.0])
        accuracy = correct / len(style_images)
        return accuracy

    def show_values(self):
        factor = "larger"
        resnet_accuracy = self.compute_style_accuracy(
            self.annotated_predictions_resnet_clip, factor=factor
        )
        vit_accuracy = self.compute_style_accuracy(
            self.annotated_predictions_vit_clip, factor=factor
        )
        print(f"{factor}: vit {vit_accuracy} vs. resnet101 {resnet_accuracy}")

        factor = "darker"
        resnet_accuracy = self.compute_style_accuracy(
            self.annotated_predictions_resnet_clip, factor=factor
        )
        vit_accuracy = self.compute_style_accuracy(
            self.annotated_predictions_vit_clip, factor=factor
        )
        print(f"{factor}: vit {vit_accuracy} vs. resnet101 {resnet_accuracy}")


if __name__ == "__main__":
    clip_vit = CLIPModelExplanations(model_name="clip_vit_b32")
    clip_resnet = CLIPModelExplanations(model_name="clip_resnet101")
    # print(clip_resnet.annotated_predictions.head())

    CompareFactorAccuracyAcrossModels().show_values()
