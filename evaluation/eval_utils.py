"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from ast import literal_eval
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def calculate_top_preds_22k(outputs, subset=True):
    # Make Mapping
    in21k_ds_mapping = pd.read_csv("../data/11_16_22_in21k_ds_mapping.csv", index_col=0)
    in21k_ds_mapping["sin21k_dsl"] = in21k_ds_mapping["sin21k_dsl"].apply(literal_eval)
    dollarstreet_id_to_dollarstreet_label = pd.Series(
        in21k_ds_mapping["sin21k_dsl"].values, index=in21k_ds_mapping["sin21k_id"]
    ).to_dict()
    combined_classes_mapping = {}
    for x in dollarstreet_id_to_dollarstreet_label.keys():
        combined_classes_mapping[x] = ",".join(dollarstreet_id_to_dollarstreet_label[x])

    # logits = outputs.logits
    adapted_logits = outputs.cpu().detach().numpy()
    if subset:
        adapted_logits = adapted_logits[:, list(in21k_ds_mapping["in21k_id"])]

    # Calculate confidences, indices, class_predictions
    confidences, indices = F.softmax(torch.from_numpy(adapted_logits), dim=-1).topk(5)

    indices = indices.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()

    # Map over indices and convert them to class strings using the indices
    class_predictions = np.vectorize(combined_classes_mapping.get)(indices)

    return indices, confidences, class_predictions


def calculate_top_preds_22k_no_subset(outputs):
    # Make Mapping
    in21k_ds_mapping = pd.read_csv("../data/11_16_22_in21k_ds_mapping.csv", index_col=0)
    in21k_ds_mapping["sin21k_dsl"] = in21k_ds_mapping["sin21k_dsl"].apply(literal_eval)
    dollarstreet_id_to_dollarstreet_label = pd.Series(
        in21k_ds_mapping["sin21k_dsl"].values, index=in21k_ds_mapping["sin21k_id"]
    ).to_dict()
    combined_classes_mapping = {}
    for x in dollarstreet_id_to_dollarstreet_label.keys():
        combined_classes_mapping[x] = ",".join(dollarstreet_id_to_dollarstreet_label[x])

    # Adapt output
    adapted_logits = outputs.cpu().detach().numpy()

    # Calculate confidences, indices, class_predictions
    confidences, indices = F.softmax(torch.from_numpy(adapted_logits), dim=-1).topk(5)

    indices = indices.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()

    # Map over indices and convert them to class strings using the indices
    class_predictions = np.vectorize(combined_classes_mapping.get)(indices)

    return indices, confidences, class_predictions


def calculate_top_preds_11k(output):
    # Make Maps
    in21k_ds_mapping = pd.read_csv("../data/11_16_22_in21k_ds_mapping.csv", index_col=0)
    in21k_ds_mapping["sin21k_dsl"] = in21k_ds_mapping["sin21k_dsl"].apply(literal_eval)
    in21k_ds_mapping = in21k_ds_mapping[in21k_ds_mapping["in21k_alibaba_id"] != -1]

    in21k_ds_mapping["sin21k_id"] = np.arange(0, 99)

    mel_dollarstreet_id_to_dollarstreet_label = pd.Series(
        in21k_ds_mapping["sin21k_dsl"].values, index=in21k_ds_mapping["sin21k_id"]
    ).to_dict()
    combined_classes_mapping = {}
    for x in mel_dollarstreet_id_to_dollarstreet_label.keys():
        combined_classes_mapping[x] = ",".join(
            mel_dollarstreet_id_to_dollarstreet_label[x]
        )

    # Take Subset
    subset_indices = in21k_ds_mapping["in21k_alibaba_id"].tolist()
    subset_100 = output[:, subset_indices]

    confidences, indices = F.softmax(subset_100, dim=-1).topk(5)

    indices = indices.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()

    # Map over indices and convert them to class strings using the indices
    class_predictions = np.vectorize(combined_classes_mapping.get)(indices)

    return indices, confidences, class_predictions


def calculate_accuracies(a):
    # Convert into lists
    a["Label_Classes"] = a["Label_Classes"].apply(literal_eval)
    a["Label_Indices"] = a["Label_Indices"].apply(literal_eval)
    a["Prediction_Classes"] = a["Prediction_Classes"].apply(lambda x: x.split("|"))
    a["Prediction_Confidences"] = a["Prediction_Confidences"].apply(
        lambda x: x.split("|")
    )
    a["Prediction_Indices"] = a["Prediction_Indices"].apply(lambda x: x.split("|"))
    a["Model_Output"] = a["Model_Output"].apply(lambda x: x.split("|"))

    a["acc5"] = a.apply(is_label_in_predictions5, axis=1)
    a["acc5"] = a["acc5"].astype(float)
    print(f"% Top 5 Accuracy {a['acc5'].sum() / len(a)}")

    a["acc1"] = a.apply(is_label_in_predictions1, axis=1)
    a["acc1"] = a["acc1"].astype(float)
    print(f"% Top 1 Accuracy {a['acc1'].sum() / len(a)}")

    a["Correct_Top5_Indices"] = a.apply(is_label_in_predictions_indices5, axis=1)
    a["Correct_Top1_Indices"] = a.apply(is_label_in_predictions_indices1, axis=1)

    # Confirm that indices accuracy matches classes accuracy
    # assert (a["acc5"] != a["Correct_Top5_Indices"]).sum() == 0
    # assert (a["acc1"] != a["Correct_Top1_Indices"]).sum() == 0

    # Drop indices accuracies
    # a = a.drop(columns=["Correct_Top5_Indices", "Correct_Top1_Indices"])

    return a


# Calculate accuracy
def is_label_in_predictions5(x):
    labels = x["Label_Classes"]  # [str]
    preds = x["Prediction_Classes"]  # ['class1,class2', 'class1,class2']

    for pred in preds[:5]:
        for pred_class in pred.split(","):
            if pred_class in labels:
                return True
    return False


def is_label_in_predictions1(x):
    labels = x["Label_Classes"]  # [str]
    preds = x["Prediction_Classes"]  # ['class1,class2', 'class1,class2']

    for pred in preds[:1]:
        for pred_class in pred.split(","):
            if pred_class in labels:
                return True
    return False


def is_label_in_predictions_indices5(x):
    labels = x["Label_Indices"]  # [int]
    preds = x["Prediction_Indices"]  # [str]
    assert type(preds[0]) == str
    assert type(labels[0]) == int

    for pred in preds[:5]:
        if int(pred) in labels:  # accounts for multilabel
            return True
    return False


def is_label_in_predictions_indices1(x):
    labels = x["Label_Indices"]  # [int]
    preds = x["Prediction_Indices"]  # [str]
    assert type(preds[0]) == str
    assert type(labels[0]) == int

    for pred in preds[:1]:
        if int(pred) in labels:  # accounts for multilabel
            return True
    return False
