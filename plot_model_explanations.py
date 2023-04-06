"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from explain_model_errors import CLIPModelExplanations, ModelExplanations, ErrorRatio
import model_accuracy_graphs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable, List, Dict, Tuple
import plotly.io as pio
import numpy as np
import os
import plotly.express as px
import pandas as pd
from scipy.stats import chisquare


def style_plot(plot_func: Callable):
    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)
        fig.update_layout(
            template="plotly_white",
            font=dict(size=14, family="Computer Modern", color="black"),
        )
        return fig

    return wrapper


class PlotMixins:
    income_bucket_to_color = {
        "low": "rgb(102, 197, 204)",
        "medium": "rgb(246, 207, 113)",
        "high": "rgb(248, 156, 116)",
    }
    factors = [
        "multiple_objects",
        "background",
        "color",
        "brighter",
        "darker",
        "larger",
        "smaller",
        "style",
        "object_blocking",
        "person_blocking",
        "partial_view",
        "pattern",
        "pose",
        "shape",
        "subcategory",
        "texture",
    ]

    colors = px.colors.qualitative.Pastel + px.colors.qualitative.Pastel1

    factor_to_color = dict(zip(factors, colors[: len(factors)]))


class PlotErrorRatio(PlotMixins):
    def __init__(self, error_ratio: ErrorRatio, save_dir: str = "plots/"):
        self.error_ratio = error_ratio
        self.save_dir = save_dir

        # prevent mathjax box upon saving
        pio.kaleido.scope.mathjax = None

    def _add_axis_labels(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(xaxis_title="factor", yaxis_title="Error ratio")
        return fig

    @style_plot
    def plot_error_ratio_total(self) -> go.Figure:
        error_ratio_total = self.error_ratio.compute_error_ratio_total()
        fig = go.Figure(
            data=go.Bar(x=error_ratio_total.index, y=error_ratio_total.values)
        )
        fig = self._add_axis_labels(fig)
        return fig

    @style_plot
    def plot_error_ratio_incomes(self) -> go.Figure:
        data = []
        for income_bucket in ["low", "medium", "high"]:
            error_ratio = self.error_ratio.compute_error_ratio_income_bucket(
                income_bucket=income_bucket
            )
            data.append(
                go.Bar(
                    name=f"{income_bucket} income",
                    x=error_ratio.index,
                    y=error_ratio.values,
                    marker_color=self.income_bucket_to_color[income_bucket],
                )
            )

        error_ratio = self.error_ratio.compute_error_ratio_total()
        data.insert(
            0,
            go.Bar(
                name="Overall",
                x=error_ratio.index,
                y=error_ratio.values,
                marker_color="#6C6C6C",
            ),
        )

        fig = go.Figure(data=data)
        fig.update_layout(barmode="group")
        fig = self._add_axis_labels(fig)
        return fig

    @style_plot
    def plot_error_ratio_regions(self) -> go.Figure:
        data = []
        for region in ["Asia", "The Americas", "Europe", "Africa"]:
            error_ratio = self.error_ratio.compute_error_ratio_region(region=region)
            data.append(go.Bar(name=region, x=error_ratio.index, y=error_ratio.values))

        error_ratio = self.error_ratio.compute_error_ratio_total()
        data.insert(
            0,
            go.Bar(
                name="Overall",
                x=error_ratio.index,
                y=error_ratio.values,
                marker_color="#6C6C6C",
            ),
        )
        fig = go.Figure(data=data)
        fig.update_layout(barmode="group")
        fig = self._add_axis_labels(fig)
        return fig

    @style_plot
    def plot_error_ratio_incomes_stacked_bar_top_5(self) -> go.Figure:
        income_buckets = ["low", "medium", "high"][::-1]
        income_bucket_to_error_ratio = {
            income_bucket: self.error_ratio.compute_error_ratio_income_bucket(
                income_bucket=income_bucket
            )[:5]
            for income_bucket in income_buckets
        }

        income_bucket_to_error_ratio[
            "overall"
        ] = self.error_ratio.compute_error_ratio_total()[:5]

        top_factors = np.unique(
            np.array(
                [
                    error_ratio_series.index
                    for error_ratio_series in income_bucket_to_error_ratio.values()
                ]
            )
        ).tolist()
        high_income_to_factor_val = self.error_ratio.compute_error_ratio_income_bucket(
            income_bucket="high"
        )
        top_factors = sorted(
            top_factors, key=lambda x: high_income_to_factor_val.to_dict()[x]
        )

        data = []

        for factor in top_factors:
            error_ratios = []
            for income_bucket in ["overall"] + income_buckets:
                error_ratio_series = income_bucket_to_error_ratio[income_bucket]
                if factor in error_ratio_series.index:
                    ratio = error_ratio_series.loc[factor]
                    if ratio < 0.0:
                        ratio = 0.0
                else:
                    ratio = 0.0
                error_ratios.append(ratio)
            # error_ratios[error_ratios < 0] = 0
            data.append(
                go.Bar(
                    name=factor.replace("_", " "),
                    x=["overall"] + income_buckets,
                    y=error_ratios,
                    marker_color=self.factor_to_color[factor],
                )
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode="stack",
            xaxis_title="Income buckets",
            yaxis_title="Error ratio",
        )
        return fig

    @style_plot
    def plot_error_ratio_regions_stacked_bar_top_5(self) -> go.Figure:
        regions = ["Asia", "The Americas", "Europe", "Africa"]
        region_to_error_ratio = {
            region: self.error_ratio.compute_error_ratio_region(region=region)[:5]
            for region in regions
        }

        region_to_error_ratio["Overall"] = self.error_ratio.compute_error_ratio_total()[
            :5
        ]

        top_factors = np.unique(
            np.array(
                [
                    error_ratio_series.index
                    for error_ratio_series in region_to_error_ratio.values()
                ]
            )
        ).tolist()
        region_to_factor_val = self.error_ratio.compute_error_ratio_region(
            region="Asia"
        )
        top_factors = sorted(
            top_factors, key=lambda x: region_to_factor_val.to_dict()[x]
        )

        data = []

        for factor in top_factors:
            error_ratios = []
            for region in ["Overall"] + regions:
                error_ratio_series = region_to_error_ratio[region]
                if factor in error_ratio_series.index:
                    ratio = error_ratio_series.loc[factor]
                    if ratio < 0.0:
                        ratio = 0.0
                else:
                    ratio = 0.0
                error_ratios.append(ratio)
            # error_ratios[error_ratios < 0] = 0
            data.append(
                go.Bar(
                    name=factor.replace("_", " "),
                    x=["Overall"] + regions,
                    y=error_ratios,
                    marker_color=self.factor_to_color[factor],
                )
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode="stack",
            xaxis_title="Regions",
            yaxis_title="Error ratio",
        )
        return fig

    def save(self):
        fig_income = self.plot_error_ratio_incomes_stacked_bar_top_5()
        pio.full_figure_for_development(fig_income, warn=False)
        save_path = os.path.join(self.save_dir, "top_5_factor_per_income_bucket.pdf")
        fig_income.write_image(save_path)
        fig_income.write_image(save_path)

        fig_region = self.plot_error_ratio_regions_stacked_bar_top_5()
        save_path = os.path.join(self.save_dir, "top_5_factor_per_region.pdf")
        fig_region.write_image(save_path)

    def save_appendix(self):
        fig_region = self.plot_error_ratio_regions()
        save_path = os.path.join(self.save_dir, "appendix_factor_per_region.pdf")
        fig_region.write_image(save_path)

        fig_region = self.plot_error_ratio_incomes()
        save_path = os.path.join(self.save_dir, "appendix_factor_per_income_bucket.pdf")
        fig_region.write_image(save_path)


class ClassDisparityExplanations:
    WORST_CLASS_REGION = [
        ("shaving", "Africa"),
        ("sofas", "Africa"),
        ("bathrooms", "Africa"),
        ("kitchen_sinks", "Africa"),
        ("showers", "Africa"),
    ]

    WORST_CLASS_INCOME_BUCKET = [
        ("sofas", "low"),
        ("toilet_paper", "low"),
        ("living_rooms", "low"),
        ("kitchen_sinks", "low"),
        ("showers", "low"),
    ]

    def __init__(self, error_ratio: ErrorRatio, save_dir: str = "plots/"):
        self.error_ratio = error_ratio
        self.save_dir = save_dir

    def show_table_explain_worst_class_region(self) -> str:
        table = self.explain_worst_class_region()
        df = pd.DataFrame(table)
        with pd.option_context("max_colwidth", 1000):
            return df.to_latex(
                index=False,
                caption="""Factors associated with model mistakes for classes in 
                    regions with the largest performance disparity. The values in parenthesis 
                    indicate how much more likely a factor is to appear for misclassified samples.
                """,
                label="tab:worst_class_region_explanation",
            )

    def show_table_explain_worst_class_income_bucket(self) -> str:
        table = self.explain_worst_class_income_bucket()
        df = pd.DataFrame(table)
        with pd.option_context("max_colwidth", 1000):
            return df.to_latex(
                index=False,
                caption="""Factors associated with model mistakes for classes in 
                    income buckets with the largest performance disparity. The values in parenthesis 
                    indicate how much more likely a factor is to appear for misclassified samples.
                """,
                label="tab:worst_class_income_bucket_explanation",
            )

    def explain_worst_class_region(self) -> List[dict]:
        table = []
        for class_label, region in self.WORST_CLASS_REGION:
            entry = dict()
            entry["Class"] = class_label
            entry["Region"] = region
            explanation = self.error_ratio.compute_error_ratio_class_region_pair(
                class_label, region
            )[:3]
            explanation_str = ", ".join(
                [f"{f} (+{r:.1f}x)" for f, r in explanation.items()]
            )
            entry["Factors associated with mistakes"] = explanation_str
            table.append(entry)
        return table

    def explain_worst_class_income_bucket(self) -> List[dict]:
        table = []
        for class_label, income in self.WORST_CLASS_INCOME_BUCKET:
            entry = dict()
            entry["Class"] = class_label
            entry["Income"] = income
            explanation = self.error_ratio.compute_error_ratio_class_income_bucket_pair(
                class_label, income
            )[:3]
            explanation_str = ", ".join(
                [f"{f} (+{r:.1f}x)" for f, r in explanation.items()]
            )
            entry["Factors associated with mistakes"] = explanation_str
            table.append(entry)
        return table


class PlotModelComparisons(PlotMixins):
    def __init__(
        self,
        model_names: Tuple[str] = ("clip_vit_b32", "clip_resnet101"),
        name_str_mapping=None,
        save_dir: str = "plots/",
        filename: str = "architecture_comparison.pdf",
    ):
        self.model_names = model_names
        self.save_dir = save_dir
        self.filename = filename

        self.model_name_to_error_ratio = self.compute_error_ratio()
        if name_str_mapping is None:
            self.name_str_mapping = {
                model_name: model_name for model_name in model_names
            }
        else:
            self.name_str_mapping = name_str_mapping

    def compute_error_ratio(self) -> Dict[str, ErrorRatio]:
        model_name_to_error_ratio = dict()
        for model_name in self.model_names:
            """
            if not model_name.startswith("clip"):
                raise ValueError(f"{model_name} data not found")

            try:
                explanations = CLIPModelExplanations(model_name=model_name)
            except AssertionError:
                explanations = CLIPModelExplanations(
                    model_name=model_name,
                    predictions_csv_dir="../data/raw/acc_per_img/",
                )
            """
            explanations = ModelExplanations(model_name=model_name)
            error_ratio = ErrorRatio(explanations)
            model_name_to_error_ratio[model_name] = error_ratio
        return model_name_to_error_ratio

    @style_plot
    def plot_error_ratio_total(self, top_n: int = 5) -> go.Figure:
        """Top_n specifies the top n factors to consider"""
        model_name_to_top_n = {
            model_name: self.model_name_to_error_ratio[
                model_name
            ].compute_error_ratio_total()[:top_n]
            for model_name in self.model_names
        }

        top_factors = np.unique(
            np.array(
                [
                    error_ratio_series.index
                    for error_ratio_series in model_name_to_top_n.values()
                ]
            )
        ).tolist()

        # used to order factors
        model_error_ratio_values = self.model_name_to_error_ratio[
            self.model_names[0]
        ].compute_error_ratio_total()

        top_factors = sorted(
            top_factors, key=lambda x: model_error_ratio_values.to_dict()[x]
        )

        data = []

        for factor in top_factors:
            error_ratios = []
            for model_name in self.model_names:
                error_ratio_series = model_name_to_top_n[model_name]
                if factor in error_ratio_series.index:
                    ratio = error_ratio_series.loc[factor]
                    if ratio < 0.0:
                        ratio = 0.0
                else:
                    ratio = 0.0
                error_ratios.append(ratio)
            # error_ratios[error_ratios < 0] = 0
            data.append(
                go.Bar(
                    name=factor.replace("_", " "),
                    x=[
                        self.name_str_mapping[model_name]
                        for model_name in self.model_names
                    ],
                    y=error_ratios,
                    marker_color=self.factor_to_color[factor],
                )
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode="stack",
            # title="Explaining CLIP mistakes",
            yaxis_title="Error ratio",
            xaxis_title="Architectures",
        )
        return fig

    def save(self):
        fig = self.plot_error_ratio_total()
        pio.full_figure_for_development(fig, warn=False)
        save_path = os.path.join(self.save_dir, self.filename)
        fig.write_image(save_path)
        fig.write_image(save_path)


class ChiSquaredTop5Factors:
    def __init__(
        self,
        model_name: str = "clip_vit_b32",
    ):
        self.model_name = model_name

        self.error_ratio = self.compute_error_ratio()

    def compute_error_ratio(self) -> ErrorRatio:
        model_name = self.model_name
        if not model_name.startswith("clip"):
            raise ValueError(f"{model_name} data not found")
        try:
            explanations = CLIPModelExplanations(model_name=model_name)
        except AssertionError:
            explanations = CLIPModelExplanations(
                model_name=model_name,
                predictions_csv_dir="../data/raw/acc_per_img/",
            )
        error_ratio = ErrorRatio(explanations)
        return error_ratio

    def compute_chi_squared(self, top_n: int = 5):
        """Top_n specifies the top n factors to consider"""
        error_ratio_top_n = self.error_ratio.compute_error_ratio_total()[:top_n]
        top_factors = error_ratio_top_n.index.tolist()

        # counts among misclassifications
        observed = self.error_ratio.annotated_predictions[
            self.error_ratio.annotated_predictions["acc5"] == 0.0
        ][top_factors].sum()

        # normalize expected to match total of observed
        expected = observed.sum() * (
            self.error_ratio.annotated_predictions[top_factors].sum()
            / self.error_ratio.annotated_predictions[top_factors].sum().sum()
        )
        result = chisquare(observed, f_exp=expected)
        return result


if __name__ == "__main__":
    clip_explanations = CLIPModelExplanations()
    error_ratio = ErrorRatio(clip_explanations)
    plot_error_ratios = PlotErrorRatio(error_ratio)
    plot_error_ratios.save()
    plot_error_ratios.save_appendix()

    model_comparison_plot = PlotModelComparisons()
    model_comparison_plot.save()

    model_comparison_all_models_plot = PlotModelComparisons(
        model_names=tuple(model_accuracy_graphs.MODEL_NAMES.keys()),
        name_str_mapping=model_accuracy_graphs.MODEL_NAMES,
        filename="all_architecture_comparison.pdf",
    )
    model_comparison_all_models_plot.save()

    chi_squared = ChiSquaredTop5Factors()
    print(chi_squared.compute_chi_squared())
