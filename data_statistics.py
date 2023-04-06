"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from annotations import Annotations
import plotly.graph_objects as go
from typing import Callable, List, Dict
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
from scipy.spatial.distance import jensenshannon
from itertools import combinations
import itertools


def style_plot(plot_func: Callable):
    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)
        fig.update_layout(
            template="plotly_white",
            yaxis_tickformat=".0%",
            font=dict(size=14, family="Computer Modern"),
        )
        return fig

    return wrapper


class PlotMixins:
    income_bucket_to_color = {
        "low": "rgb(102, 197, 204)",
        "medium": "rgb(246, 207, 113)",
        "high": "rgb(248, 156, 116)",
    }


class DollarStreetStats:
    def __init__(self, annotations: Annotations, save_dir: str = "plots/"):
        self.annotations = annotations
        self.table = self.annotations.annotations
        self.save_dir = save_dir

    @property
    def num_images_per_class(self):
        return self.table["class"].value_counts()

    @property
    def num_images_per_region(self):
        return self.table.region.value_counts()

    @property
    def num_images_per_income_bucket(self):
        return self.table.income_bucket.value_counts()

    @property
    def num_households(self):
        return len(self.table.household_id.unique())

    @style_plot
    def plot_dist_num_images_per_class(self) -> go.Figure:
        x = np.sort(self.num_images_per_class.values)
        fig = go.Figure()
        fig.add_trace(go.Box(x=x, name="images per class"))
        return fig

    def save_plots(self):
        num_images_per_class_fig = self.plot_dist_num_images_per_class()
        pio.full_figure_for_development(num_images_per_class_fig, warn=False)
        save_path = os.path.join(self.save_dir, "num_images_per_class.pdf")
        num_images_per_class_fig.write_image(save_path)


class FactorStats(PlotMixins):
    def __init__(self, annotations: Annotations, save_dir: str = "plots/"):
        self.annotations = annotations
        self.table = self.annotations.annotations
        self.save_dir = save_dir
        self.regions = self.table.region.unique().tolist()
        self.income_buckets = ["low", "medium", "high"]

    @style_plot
    def plot_factor_distribution_by_region(self) -> go.Figure:
        """Plots"""
        data = []

        for region in self.regions:
            table = self.table[self.table["region"] == region]
            percents = table[self.annotations.factors].sum() / table.shape[0]
            data.append(
                go.Bar(x=self.annotations.factors, y=percents.values, name=region)
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            # title="Factor distribution by geography",
            xaxis={"categoryorder": "total descending"},
            yaxis_title="percent of images",
            barmode="group",
        )

        return fig

    @style_plot
    def plot_factor_distribution_by_income(self) -> go.Figure:
        """Plots"""
        data = []

        for income_bucket in self.income_buckets:
            table = self.table[self.table["income_bucket"] == income_bucket]
            percents = table[self.annotations.factors].sum() / table.shape[0]
            data.append(
                go.Bar(
                    x=self.annotations.factors,
                    y=percents.values,
                    name=income_bucket,
                    marker_color=self.income_bucket_to_color[income_bucket],
                )
            )

        fig = go.Figure(data=data)
        fig.update_layout(
            # title=f"Factor distribution by income bucket",
            xaxis={"categoryorder": "total descending"},
            yaxis_title="percent of images",
            barmode="group",
        )

        return fig

    def save_plots(self):
        region_factors = self.plot_factor_distribution_by_region()
        pio.full_figure_for_development(region_factors, warn=False)
        save_path = os.path.join(self.save_dir, "region_factors.pdf")
        region_factors.write_image(save_path)

        income_factors = self.plot_factor_distribution_by_income()
        pio.full_figure_for_development(income_factors, warn=False)
        save_path = os.path.join(self.save_dir, "income_factors.pdf")
        income_factors.write_image(save_path)


class FactorStatsByClass:
    """Computes statistics of factors by class.

    Preprocessing removes regions or incomes for class containing fewer than 5 images.
    This results in the removal of 17 income/class combinations
        and 17 region/class combinations.
    """

    def __init__(self, annotations: Annotations, save_dir: str = "plots/"):
        self.annotations = annotations
        self.table = self.annotations.annotations
        self.save_dir = save_dir
        self.regions = self.table.region.unique().tolist()
        self.income_buckets = ["low", "medium", "high"]

        self.class_factor_by_income_bucket = self.build_normalized_factors_by_group(
            group="income_bucket"
        )
        self.class_factor_by_region = self.build_normalized_factors_by_group(
            group="region"
        )

        self.pairwise_distance_by_income = pd.DataFrame(
            self.compute_pairwise_group_distances()
        ).sort_values(by=["dist"], ascending=False)
        self.pairwise_distance_by_region = pd.DataFrame(
            self.compute_pairwise_group_distances(group="region")
        ).sort_values(by=["dist"], ascending=False)

    def build_normalized_factors_by_group(
        self, group: str = "income_bucket"
    ) -> pd.DataFrame:
        """
        Builds a normalized vector of factor counts for each class per group.

        Args:
            group: can be income_bucket or region
        """
        factors = self.annotations.factors
        factors_plus_class = factors + ["class", group]
        grouped_table = self.table[factors_plus_class].groupby(["class", group]).sum()
        normalized = grouped_table.divide(grouped_table.sum(axis=1), axis="index")
        normalized["number_of_images"] = (
            self.table[factors_plus_class]
            .groupby(["class", group])
            .count()["multiple_objects"]
        )

        # filter class, group combinations with fewer than 5 images
        normalized = normalized[normalized["number_of_images"] >= 5].drop(
            columns="number_of_images"
        )
        return normalized

    def compute_pairwise_group_distances(
        self, group: str = "income_bucket"
    ) -> List[Dict]:
        """Computes the Shannon Jensen divergence between the factor distributions per class
        across each pair of groups.

        Args:
            group: can be income_bucket or region

        Returns: list of dictionaries, each containing "class", "dist", "income1", "income2"
        """
        distances = []

        buckets = self.regions if group == "region" else self.income_buckets
        factor_distributions = getattr(self, f"class_factor_by_{group}")

        for class_name in factor_distributions.index.get_level_values(0).unique():
            for bucket_pair in list(combinations(buckets, 2)):
                bucket1 = bucket_pair[0]
                bucket2 = bucket_pair[1]
                try:
                    dist1 = factor_distributions.loc[class_name, bucket1]
                    dist2 = factor_distributions.loc[class_name, bucket2]
                except KeyError:
                    continue

                dist = jensenshannon(dist1, dist2)
                entry = {
                    "class": class_name,
                    f"{group}1": bucket1,
                    f"{group}2": bucket2,
                    "dist": dist,
                }
                entry.update(self.get_distinctive_factors(dist1, dist2))
                distances.append(entry)
        return distances

    def pairwise_distance_to_latex(
        self, group: str = "income_bucket", show_most_different: bool = True
    ) -> str:
        """Returns the largest (or smallest) pairwise difference across groups.

        Args:
            group: income_bucket or region
            show_most_different: if true, return the largest differences. Otherwise the least different groups
        """
        table = (
            self.pairwise_distance_by_region
            if group == "region"
            else self.pairwise_distance_by_income
        )
        table = table.copy()

        table[f"{group}s"] = table[[f"{group}1", f"{group}2"]].apply(
            " vs. ".join, axis=1
        )

        top_factor_names = [f"top_{i}_distinctive_factor" for i in range(1, 4)]
        # perc_diff_cols_names = [
        #     f"top_{i}_distinctive_factor_perc_diff" for i in range(1, 4)
        # ]
        # factor_and_perc_diff_names = list(
        #     itertools.chain(*zip(top_factor_names, perc_diff_cols_names))
        # )

        table["distinctive factors"] = table[top_factor_names].agg(", ".join, axis=1)

        cols = ["class", f"{group}s", "distinctive factors"]
        ascending = False if show_most_different else True
        sorted_table = table.sort_values(
            by=["dist", "class"], ascending=ascending
        ).head(10)
        result = sorted_table[cols].to_latex(index=False)
        return result

    def get_distinctive_factors(self, dist1, dist2, top_n: int = 3) -> dict:
        factors = {}

        i = 1
        for n, v in (dist1 - dist2).abs().nlargest(3).items():
            factors[f"top_{i}_distinctive_factor"] = n
            factors[f"top_{i}_distinctive_factor_perc_diff"] = v
            i += 1
        return factors


if __name__ == "__main__":
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    # first write to avoid mathjax warning in saved plots
    fig.write_image(os.path.join("plots/", "test.pdf"))

    labels = Annotations()
    dollarstreet_stats = DollarStreetStats(annotations=labels)
    dollarstreet_stats.save_plots()

    factor_stats = FactorStats(annotations=labels)
    factor_stats.save_plots()
