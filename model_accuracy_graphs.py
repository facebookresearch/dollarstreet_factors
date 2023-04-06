"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict

import pandas as pd

import graph_geography
import plotly.graph_objs as go
import plotly.express as px
from annotations import Annotations
import data_statistics
from explain_model_errors import CLIPModelExplanations, ModelExplanations

PLOTS_DIR = "plots"

MODEL_NAMES = OrderedDict(
    clip_vit_b32="CLIP ViT B/32",
    clip_resnet101="CLIP ResNet101",
    ViTPretrained21k="ViT B/16 IN21k",
    ResNet50Pretrained21k="ResNet50 IN21k",
    BEiTPretrained21k="BEiT L/16 IN21k",
    MLPMixerPretrained21k="MLP Mixer B/16 IN21k",
)


class PlotMixins:
    income_bucket_to_color = {
        "low": "rgb(102, 197, 204)",
        "medium": "rgb(246, 207, 113)",
        "high": "rgb(248, 156, 116)",
    }


def graph_income_accuracy(
    model="clip_vit_b32", save=False, filename="income_bucket_accuracy.pdf"
):
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()
    per_income = per_image.groupby("income_bucket").mean()
    per_income = per_income.loc[["high", "medium", "low"]]
    per_income = per_income.reset_index()
    fig = px.bar(
        per_income,
        x="income_bucket",
        y="acc5",
        labels={
            "income_bucket": "Income Bucket",
            "acc5": "Top 5 Accuracy",
        },
        color="income_bucket",  # if values in column z = 'some_group' and 'some_other_group'
        color_discrete_map=PlotMixins.income_bucket_to_color,
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_income_accuracy_scatter(
    model="clip_vit_b32", save=False, filename="income_accuracy_scatter.pdf"
):
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()
    per_income_point = per_image.groupby("income").mean().reset_index()
    fig = px.scatter(
        per_income_point,
        x="income",
        y="acc5",
        labels={
            "income": "Income",
            "acc5": "Top 5 Accuracy",
        },
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )

    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_top_5_disparity_classes_region(
    model="clip_vit_b32", save=False, filename="region_accuracy_differences.pdf"
):
    """
    1) Take max per class
    2) max - accuracy per region
    3) Take top 10
    """
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()
    cls_region = per_image.groupby(["class", "region"]).mean()

    cls_max_region = cls_region.groupby("class").max()
    cls_max_region = cls_max_region["acc5"]
    cls_region_with_max = cls_region.join(
        cls_max_region, on="class", rsuffix="_max_region"
    )
    cls_region_with_max["diff"] = (
        cls_region_with_max["acc5_max_region"] - cls_region_with_max["acc5"]
    )
    cls_region_with_max = cls_region_with_max.sort_values("diff", ascending=False)

    classes = set()
    target_n = 10
    while len(classes) < 10:
        classes = set(cls_region_with_max.iloc[0:target_n].reset_index()["class"])
        target_n += 1
    region_accuracy_disparities = (
        cls_region.loc[classes].reset_index().set_index("region")
    )
    trace1 = go.Bar(
        x=region_accuracy_disparities.loc["Africa"]["class"],
        y=region_accuracy_disparities.loc["Africa"].acc5,
        name="Africa",
    )
    trace2 = go.Bar(
        x=region_accuracy_disparities.loc["Asia"]["class"],
        y=region_accuracy_disparities.loc["Asia"].acc5,
        name="Asia",
    )
    trace3 = go.Bar(
        x=region_accuracy_disparities.loc["Europe"]["class"],
        y=region_accuracy_disparities.loc["Europe"].acc5,
        name="Europe",
    )
    trace4 = go.Bar(
        x=region_accuracy_disparities.loc["The Americas"]["class"],
        y=region_accuracy_disparities.loc["The Americas"].acc5,
        name="The Americas",
    )
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(barmode="group", yaxis_title="Top 5 Accuracy")
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_top_5_disparity_classes_region_5_or_more(
    model="clip_vit_b32",
    save=False,
    filename="region_accuracy_differences_5_or_more.pdf",
):
    """
    1) Take max per class
    2) max - accuracy per region
    3) Take top 10
    """
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()

    cls_region_count = per_image.groupby(["class", "region"]).count()
    cls_count_regions = cls_region_count.groupby("class").count()
    too_few_regions = cls_count_regions[cls_count_regions["acc1"] < 4]

    min_cls_region_count = cls_region_count.groupby("class").min()
    too_few_samples = min_cls_region_count[min_cls_region_count["acc1"] < 5]

    classes_to_ignore = set(too_few_samples.index).union(set(too_few_regions.index))

    per_image["keep"] = per_image.apply(
        lambda x: x["class"] not in classes_to_ignore, axis=1
    )
    per_image = per_image[per_image["keep"]]

    cls_region = per_image.groupby(["class", "region"]).mean()

    cls_max_region = cls_region.groupby("class").max()
    cls_max_region = cls_max_region["acc5"]
    cls_region_with_max = cls_region.join(
        cls_max_region, on="class", rsuffix="_max_region"
    )
    cls_region_with_max["diff"] = (
        cls_region_with_max["acc5_max_region"] - cls_region_with_max["acc5"]
    )
    cls_region_with_max = cls_region_with_max.sort_values("diff", ascending=False)

    classes = set()
    target_n = 10
    while len(classes) < 10:
        classes = set(cls_region_with_max.iloc[0:target_n].reset_index()["class"])
        target_n += 1
    print(cls_region_count.loc[list(classes)]["acc1"])
    region_accuracy_disparities = (
        cls_region.loc[classes].reset_index().set_index("region")
    )
    trace1 = go.Bar(
        x=region_accuracy_disparities.loc["Africa"]["class"],
        y=region_accuracy_disparities.loc["Africa"].acc5,
        name="Africa",
    )
    trace2 = go.Bar(
        x=region_accuracy_disparities.loc["Asia"]["class"],
        y=region_accuracy_disparities.loc["Asia"].acc5,
        name="Asia",
    )
    trace3 = go.Bar(
        x=region_accuracy_disparities.loc["Europe"]["class"],
        y=region_accuracy_disparities.loc["Europe"].acc5,
        name="Europe",
    )
    trace4 = go.Bar(
        x=region_accuracy_disparities.loc["The Americas"]["class"],
        y=region_accuracy_disparities.loc["The Americas"].acc5,
        name="The Americas",
    )
    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(barmode="group", yaxis_title="Top 5 Accuracy")
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_top_5_disparity_classes_income(
    model="clip_vit_b32", save=False, filename="income_accuracy_differences.pdf"
):
    """
    1) Take max per class
    2) max - accuracy per income
    3) Take top 10
    """
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()
    cls_income = per_image.groupby(["class", "income_bucket"]).mean()

    cls_max_income = cls_income.groupby("class").max()
    cls_max_income = cls_max_income["acc5"]
    cls_income_with_max = cls_income.join(
        cls_max_income, on="class", rsuffix="_max_income"
    )
    cls_income_with_max["diff"] = (
        cls_income_with_max["acc5_max_income"] - cls_income_with_max["acc5"]
    )
    cls_income_with_max = cls_income_with_max.sort_values("diff", ascending=False)

    classes = set()
    target_n = 10
    while len(classes) < 10:
        classes = set(cls_income_with_max.iloc[0:target_n].reset_index()["class"])
        target_n += 1

    income_accuracy_disparities = (
        cls_income.loc[classes].reset_index().set_index("income_bucket")
    )
    trace3 = go.Bar(
        x=income_accuracy_disparities.loc["high"]["class"],
        y=income_accuracy_disparities.loc["high"].acc5,
        name="$High$",
        marker_color=PlotMixins.income_bucket_to_color["high"],
    )
    trace1 = go.Bar(
        x=income_accuracy_disparities.loc["low"]["class"],
        y=income_accuracy_disparities.loc["low"].acc5,
        name="$Low$",
        marker_color=PlotMixins.income_bucket_to_color["low"],
    )
    trace2 = go.Bar(
        x=income_accuracy_disparities.loc["medium"]["class"],
        y=income_accuracy_disparities.loc["medium"].acc5,
        name="$Medium$",
        marker_color=PlotMixins.income_bucket_to_color["medium"],
    )
    data = [trace3, trace2, trace1]
    layout = go.Layout(barmode="group", yaxis_title="Top 5 Accuracy")
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_top_5_disparity_classes_income_limit_5_or_more(
    model="clip_vit_b32",
    save=False,
    filename="income_accuracy_differences_5_or_more.pdf",
):
    """
    1) Take max per class
    2) max - accuracy per income
    3) Take top 10
    """
    predictions = CLIPModelExplanations(model_name=model).annotated_predictions
    per_image = predictions.groupby("full_image_id").first()
    cls_income_count = per_image.groupby(["class", "income_bucket"]).count()
    cls_count_incomes = cls_income_count.groupby("class").count()
    too_few_incomes = cls_count_incomes[cls_count_incomes["acc1"] < 3]

    min_cls_income_count = cls_income_count.groupby("class").min()
    too_few_samples = min_cls_income_count[min_cls_income_count["acc1"] < 5]

    classes_to_ignore = set(too_few_samples.index).union(set(too_few_incomes.index))

    per_image["keep"] = per_image.apply(
        lambda x: x["class"] not in classes_to_ignore, axis=1
    )
    per_image = per_image[per_image["keep"]]
    cls_income = per_image.groupby(["class", "income_bucket"]).mean()

    cls_max_income = cls_income.groupby("class").max()
    cls_max_income = cls_max_income["acc5"]
    cls_income_with_max = cls_income.join(
        cls_max_income, on="class", rsuffix="_max_income"
    )
    cls_income_with_max["diff"] = (
        cls_income_with_max["acc5_max_income"] - cls_income_with_max["acc5"]
    )
    cls_income_with_max = cls_income_with_max.sort_values("diff", ascending=False)

    classes = set()
    target_n = 10
    while len(classes) < 10:
        classes = set(cls_income_with_max.iloc[0:target_n].reset_index()["class"])
        target_n += 1
    print(cls_income_count.loc[list(classes)]["acc1"])
    income_accuracy_disparities = (
        cls_income.loc[classes].reset_index().set_index("income_bucket")
    )
    trace3 = go.Bar(
        x=income_accuracy_disparities.loc["high"]["class"],
        y=income_accuracy_disparities.loc["high"].acc5,
        name="High",
        marker_color=PlotMixins.income_bucket_to_color["high"],
    )
    trace1 = go.Bar(
        x=income_accuracy_disparities.loc["low"]["class"],
        y=income_accuracy_disparities.loc["low"].acc5,
        name="Low",
        marker_color=PlotMixins.income_bucket_to_color["low"],
    )
    trace2 = go.Bar(
        x=income_accuracy_disparities.loc["medium"]["class"],
        y=income_accuracy_disparities.loc["medium"].acc5,
        name="Medium",
        marker_color=PlotMixins.income_bucket_to_color["medium"],
    )
    data = [trace3, trace2, trace1]
    layout = go.Layout(barmode="group", yaxis_title="Top 5 Accuracy")
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_jsd_quartile_accuracy_gap_income(
    model="clip_vit_b32",
    save=False,
    group="income",
    filename="jsd_quartile_accuracy_gap_income.pdf",
):

    factors = data_statistics.FactorStatsByClass(Annotations())
    predictions = ModelExplanations(model_name=model).annotated_predictions

    predictions_per_image = predictions.groupby("full_image_id").first()
    acc_class_income = predictions_per_image.groupby(["class", "income_bucket"]).mean()

    def diff(x, income):
        acc1 = x["acc5"]
        try:
            acc2 = acc_class_income.loc[[(x.name[0], income)]]["acc5"].item()
        except:
            acc2 = acc1
            return abs(acc1 - acc2)
        return abs(acc1 - acc2)

    acc_class_income["acc_diff_high"] = acc_class_income.apply(
        lambda x: diff(x, "high"), axis=1
    )
    acc_class_income["acc_diff_medium"] = acc_class_income.apply(
        lambda x: diff(x, "medium"), axis=1
    )
    acc_class_income["acc_diff_low"] = acc_class_income.apply(
        lambda x: diff(x, "low"), axis=1
    )

    jsd_accuracies = []
    for each in factors.compute_pairwise_group_distances():
        cls = each["class"]
        i1 = each["income_bucket1"]
        i2 = each["income_bucket2"]
        j_dist = each["dist"]
        row = acc_class_income.loc[(cls, i1)]
        a_dis = row[f"acc_diff_{i2}"].item()
        jsd_accuracies.append(
            {
                "class": cls,
                "income_bucket1": i1,
                "income_bucket2": i2,
                "jsd_distance": j_dist,
                "acc_difference": a_dis,
            }
        )

    jsd_accuracies_df = pd.DataFrame(jsd_accuracies)
    thresholds = jsd_accuracies_df.quantile([0.0, 0.25, 0.5, 0.75, 1.0])[
        "jsd_distance"
    ].tolist()
    percentile_means = {}
    for i in range(1, len(thresholds)):
        perc = 0.25 * i
        values = jsd_accuracies_df[
            (jsd_accuracies_df["jsd_distance"] < thresholds[i] + 1e-6)
            & (jsd_accuracies_df["jsd_distance"] >= thresholds[i - 1])
        ]
        mean_j = values["jsd_distance"].mean()
        mean_a = values["acc_difference"].mean()
        percentile_means[str(100 * perc) + "%"] = {"acc_difference": mean_a}

    fig = px.bar(
        pd.DataFrame(percentile_means).transpose(),
        y="acc_difference",
    )
    fig.update_layout(
        xaxis={"type": "category", "title": "JSD Percentile"},
        yaxis={"title": "Difference in accuracy between<br>income bucket pairs"},
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_jsd_quartile_accuracy_gap_region(
    model="clip_vit_b32",
    save=False,
    group="income",
    filename="jsd_quartile_accuracy_gap_region.pdf",
):

    factors = data_statistics.FactorStatsByClass(Annotations())
    predictions = ModelExplanations(model_name=model).annotated_predictions

    predictions_per_image = predictions.groupby("full_image_id").first()
    acc_class_region = predictions_per_image.groupby(["class", "region"]).mean()

    def diff(x, region):
        acc1 = x["acc5"]
        try:
            acc2 = acc_class_region.loc[[(x.name[0], region)]]["acc5"].item()
        except:
            acc2 = acc1
            return abs(acc1 - acc2)
        return abs(acc1 - acc2)

    acc_class_region["acc_diff_Asia"] = acc_class_region.apply(
        lambda x: diff(x, "Asia"), axis=1
    )
    acc_class_region["acc_diff_Africa"] = acc_class_region.apply(
        lambda x: diff(x, "Africa"), axis=1
    )
    acc_class_region["acc_diff_Europe"] = acc_class_region.apply(
        lambda x: diff(x, "Europe"), axis=1
    )
    acc_class_region["acc_diff_The Americas"] = acc_class_region.apply(
        lambda x: diff(x, "The Americas"), axis=1
    )

    jsd_accuracies = []
    for each in factors.compute_pairwise_group_distances(group="region"):
        cls = each["class"]
        r1 = each["region1"]
        r2 = each["region2"]
        j_dist = each["dist"]
        row = acc_class_region.loc[(cls, r1)]
        a_dis = row[f"acc_diff_{r2}"].item()
        jsd_accuracies.append(
            {
                "class": cls,
                "region1": r1,
                "region2": r2,
                "jsd_distance": j_dist,
                "acc_difference": a_dis,
            }
        )

    jsd_accuracies_df = pd.DataFrame(jsd_accuracies)
    thresholds = jsd_accuracies_df.quantile([0.0, 0.25, 0.5, 0.75, 1.0])[
        "jsd_distance"
    ].tolist()
    percentile_means = {}
    for i in range(1, len(thresholds)):
        perc = 0.25 * i
        values = jsd_accuracies_df[
            (jsd_accuracies_df["jsd_distance"] < thresholds[i] + 1e-6)
            & (jsd_accuracies_df["jsd_distance"] >= thresholds[i - 1])
        ]
        mean_j = values["jsd_distance"].mean()
        mean_a = values["acc_difference"].mean()
        percentile_means[str(100 * perc) + "%"] = {"acc_difference": mean_a}

    fig = px.bar(
        pd.DataFrame(percentile_means).transpose(),
        y="acc_difference",
    )
    fig.update_layout(
        xaxis={"type": "category", "title": "JSD Percentile"},
        yaxis={"title": "Difference in accuracy between<br>region pairs"},
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_income_vit_resnet(save=False, filename="income_accuracy_vit_resnet.pdf"):
    predictions_vit = CLIPModelExplanations().annotated_predictions
    predictions_res = CLIPModelExplanations("clip_resnet101").annotated_predictions
    per_image_vit = predictions_vit.groupby("full_image_id").first()
    per_image_vit = per_image_vit.groupby("income_bucket").mean()
    per_image_vit = per_image_vit.loc[["high", "medium", "low"]]
    per_image_vit = per_image_vit.reset_index()

    per_image_res = predictions_res.groupby("full_image_id").first()
    per_image_res = per_image_res.groupby("income_bucket").mean()
    per_image_res = per_image_res.loc[["high", "medium", "low"]]
    per_image_res = per_image_res.reset_index()

    fig = go.Figure(
        data=[
            go.Bar(
                name="ViT B/32",
                x=per_image_vit["income_bucket"],
                y=per_image_vit["acc5"],
            ),
            go.Bar(
                name="ResNet101",
                x=per_image_res["income_bucket"],
                y=per_image_res["acc5"],
            ),
        ]
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
        yaxis_title="Top 5 Accuracy",
        xaxis_title="Income Bucket",
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_region_vit_resnet(save=False, filename="region_accuracy_vit_resnet.pdf"):
    predictions_vit = CLIPModelExplanations().annotated_predictions
    predictions_res = CLIPModelExplanations("clip_resnet101").annotated_predictions
    per_image_vit = predictions_vit.groupby("full_image_id").first()
    per_image_vit = per_image_vit.groupby("region").mean()
    per_image_vit = per_image_vit.reset_index()

    per_image_res = predictions_res.groupby("full_image_id").first()
    per_image_res = per_image_res.groupby("region").mean()
    per_image_res = per_image_res.reset_index()

    fig = go.Figure(
        data=[
            go.Bar(
                name="ViT B/32",
                x=per_image_vit["region"],
                y=per_image_vit["acc5"],
            ),
            go.Bar(
                name="ResNet101",
                x=per_image_res["region"],
                y=per_image_res["acc5"],
            ),
        ]
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
        yaxis_title="Top 5 Accuracy",
        xaxis_title="Region",
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_income_model_comparison(
    save=False, filename="income_accuracy_model_comparison.pdf"
):
    plotting_data = []
    for model in MODEL_NAMES:
        predictions = ModelExplanations(model_name=model).annotated_predictions
        per_image = predictions.groupby("full_image_id").first()
        per_image = per_image.groupby("income_bucket").mean()
        per_image = per_image.loc[["high", "medium", "low"]]
        per_image = per_image.reset_index()
        plotting_data.append(
            go.Bar(
                name=MODEL_NAMES[model],
                x=per_image["income_bucket"],
                y=per_image["acc5"],
            )
        )

    fig = go.Figure(data=plotting_data)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
        yaxis_title="Top 5 Accuracy",
        xaxis_title="Income Bucket",
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")


def graph_region_model_comparison(
    save=False, filename="region_accuracy_model_comparison.pdf"
):
    plotting_data = []
    for model in MODEL_NAMES:
        predictions = ModelExplanations(model_name=model).annotated_predictions
        per_image = predictions.groupby("full_image_id").first()
        per_image = per_image.groupby("region").mean()
        per_image = per_image.reset_index()
        plotting_data.append(
            go.Bar(
                name=MODEL_NAMES[model],
                x=per_image["region"],
                y=per_image["acc5"],
            )
        )

    fig = go.Figure(data=plotting_data)
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat=".0%",
        font=dict(family="Serif", size=14),
        yaxis_title="Top 5 Accuracy",
        xaxis_title="Income Bucket",
    )
    fig.show()
    if save:
        fig.write_image(f"{PLOTS_DIR}/{filename}")
