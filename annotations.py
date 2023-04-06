"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
Module to read annotations
"""
import ast
import random
import re
import pandas as pd
import json
from typing import Dict, Mapping, Optional, List
from pathlib import Path
import os
from sklearn.preprocessing import MultiLabelBinarizer
import spacy
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

FACTORS = [
    "multiple_objects",
    "background",
    "color",
    "brighter",
    "darker",
    "style",
    "larger",
    "smaller",
    "object_blocking",
    "person_blocking",
    "partial_view",
    "pattern",
    "pose",
    "shape",
    "subcategory",
    "texture",
]

AUGMENTED_FACTORS = {
    "multiple_objects": "multiple numerous many objects",
    "background": "background environment scene setting surrounding",
    "color": "color complexion hue tone pigmentation",
    "brighter": "brighter lighter vivid glowing flashing",
    "darker": "darker dim shadow poorly lit",
    "style": "style",
    "larger": "larger bigger taller broad large scale closer",
    "smaller": "smaller compact tiny miniature little farther",
    "object_blocking": "object blocking obstruct occlusion",
    "person_blocking": "person blocking obstruct occlusion",
    "partial_view": "partial view cropped out of frame clipped",
    "pattern": "pattern design ornamentation",
    "pose": "pose position posture stance perspective view",
    "shape": "shape figure form outline",
    "subcategory": "subcategory variety brand model",
    "texture": "texture finish consistency feel",
}


class Annotations:
    """Builds a dataframe containing the annotation results.
    Args:
        data_dir (str): containing the path to the annotation csv files.
    Attributes:
        annotations (DataFrame): contains annotations and metadata
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.factors = FACTORS

        self.annotations_all: pd.DataFrame = self.read_annotations()
        self.annotations: pd.DataFrame = self.annotations_all[
            self.annotations_all["can_i_rate_this_job"] != "no"
        ]

    def read_annotations(self):
        file_path = os.path.join(self.data_dir, "annotations.json")
        df = pd.read_json(file_path, convert_axes=False, orient="split")
        df.index.name = "task_media_id"
        return df


class Preprocess:
    """Cleans and combines annotations with metadata"""

    def __init__(self, data_dir: str = "data/raw/"):
        self.data_dir = data_dir

        self.annotations_raw: pd.DataFrame = self.read_annotations()
        self.task_metadata: pd.DataFrame = self.read_task_metadata()
        self.image_metadata: pd.DataFrame = self.read_image_metadata()

        self.annotations: pd.DataFrame = self.merge_metadata()

    def merge_metadata(self) -> pd.DataFrame:
        df = self.annotations_raw.join(self.task_metadata)
        df = self.add_dollar_street_metadata(df)
        return df

    def read_task_metadata(self):
        file_path = os.path.join(
            self.data_dir,
            "raw_annotation_tasks.csv",
        )
        df = pd.read_csv(
            file_path,
            dtype={
                "job_id": str,
                "media_id": str,
                "url": str,
                "class": str,
                "region": str,
            },
            index_col=0,
        )
        df = df.rename(columns={"media_id": "task_media_id"})
        df["image_id"] = df["url"].apply(Preprocess.parse_image_id)
        df = df.set_index("task_media_id")
        df["income_bucket"] = df["income_bucket"].map(
            {1: "low", 2: "medium", 3: "high"}
        )
        cols_to_keep = ["class", "url", "image_id", "region", "income_bucket"]
        df = df[cols_to_keep]
        return df

    @staticmethod
    def parse_image_id(image_url: str) -> str:
        """Parses id from url"""
        file_name = image_url.split("/")[-1]
        file_name = file_name.replace(".jpg", "").replace(".jpeg", "")
        file_name = file_name.replace("480x480-", "").replace("-", "")
        return file_name

    def read_image_metadata(self) -> pd.DataFrame:
        """Reads metadata from
        https://dl.fbaipublicfiles.com/vissl/fairness/geographical_diversity/metadata_full_dollar_street.json
        """
        file_path = os.path.join(self.data_dir, "metadata_full_dollar_street.json")
        meta_df = pd.read_json(
            file_path, dtype={"url": str, "country": str, "income": float}
        )
        meta_df = meta_df[["url", "income", "country", "lat", "lng", "id"]]
        meta_df["household_id"] = meta_df.groupby(["country", "income"]).ngroup()
        meta_df = meta_df.set_index("url")
        meta_df = meta_df.rename(columns={"id": "full_image_id"})
        # drop duplicated urls
        meta_df = meta_df.loc[~meta_df.index.duplicated()]
        return meta_df

    def add_dollar_street_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds metadata from
        https://dl.fbaipublicfiles.com/vissl/fairness/geographical_diversity/metadata_full_dollar_street.json
        """
        return df.join(self.image_metadata, on="url", how="left")

    def save(self) -> None:
        """Saves preprocessed annotations"""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(self.data_dir)), "annotations.json"
        )
        self.annotations.to_json(file_path, orient="split")

    def read_annotations(self) -> pd.DataFrame:
        annotations_path = os.path.join(
            self.data_dir,
            "annotations_raw.csv",
        )

        df = pd.read_csv(annotations_path, dtype={"job_id": str, "media_id": str})
        df = df.rename(columns={"media_id": "task_media_id"})
        df = df.set_index("task_media_id")
        df = df.loc[~df.index.duplicated(keep="first")].reset_index()
        df = self._drop_empty_rows(df)
        df = df.rename(columns={"augmentation": "justification"})

        df = self.explode_categories(df)
        df.drop(
            labels="select_all_categories_right_different_from_left",
            inplace=True,
            axis=1,
        )

        df = self.rename_factors(df)
        df = self.cleanup_one_word(df)
        df = df.set_index("task_media_id")
        return df

    def _drop_empty_rows(self, df) -> pd.DataFrame:
        return df[df["select_all_categories_right_different_from_left"].notna()]

    def explode_categories(self, df) -> pd.DataFrame:
        mlb = MultiLabelBinarizer()
        series = df["select_all_categories_right_different_from_left"].apply(
            ast.literal_eval
        )
        values = mlb.fit_transform(series)
        self.categories = mlb.classes_
        var_axes = pd.DataFrame(values, columns=mlb.classes_, index=df.index)
        return pd.concat([df, var_axes], axis=1)

    def rename_factors(self, df) -> pd.DataFrame:
        # cleaning up
        rename_map = {
            "another_object_is_present": "multiple_objects",
            "background": "background",
            "color": "color",
            "lighting_is_brighter": "brighter",
            "lighting_is_darker": "darker",
            "media_style": "style",
            "object_is_large": "larger",
            "object_is_small": "smaller",
            "object_partially_blocked_by_another_object": "object_blocking",
            "object_partially_blocked_by_person": "person_blocking",
            "object_partially_present": "partial_view",
            "pattern": "pattern",
            "pose_positioning": "pose",
            "shape": "shape",
            "subcategory": "subcategory",
            "texture": "texture",
            "subject_location": "location",
            #
            "true class": "class",
            "1_word": "one_word",
        }
        self.categories = [rename_map[cat] for cat in self.categories]
        df.rename(columns=rename_map, inplace=True)
        return df

    def cleanup_one_word(self, df) -> pd.DataFrame:
        patt = re.compile(r"[\w']+")
        df["one_word"] = (
            df["one_word"].str.lower().dropna().apply(patt.findall).str.join(" ")
        )
        return df


class TopFactorBuilder:
    """Compute the prevalence of the top factor.
    Top factor is based on the text justification
    This takes ~10 minutes since computing word embedding similarities is costly
    """

    def __init__(self, annotations):
        self.annotations = annotations

        # Spacy pre-trained embedding model
        self.embed = spacy.load("en_core_web_lg")
        self.factors = FACTORS
        self.factor_to_vec = {f: self.embed(AUGMENTED_FACTORS[f]) for f in self.factors}

        self.top_factor_df = self.build_top_factor_df()

    def build_top_factor_df(self) -> pd.DataFrame:
        """Selected the top factor based on embedding similarity to the justification"""
        df = self.annotations.copy()
        df[self.factors] = 0

        selected_factors = self.annotations.apply(self._find_selected_factors, axis=1)

        def _find_top_factor(x: pd.Series) -> str:
            justifications = [x.justification, x.one_word]
            justifications = [
                w.lower().strip() for w in justifications if w and (w == w)
            ]
            return self.find_top_factor(
                selected_factors.loc[x.name], " ".join(justifications)
            )

        top_factor = df.progress_apply(_find_top_factor, axis=1)

        for factor in self.factors:
            df[factor] = top_factor == factor
            df[factor] = df[factor].astype(int)
        return df

    def write(self, path):
        self.top_factor_df.to_json(path, orient="records", lines=True)

    def find_top_factor(self, factors: List[str], justification: str) -> Optional[str]:
        if not factors:
            return None

        if not justification:
            return random.choice(factors)

        justification_v = self.embed(self.process_text(justification))

        # Sometimes justifications will have only stop words in them or words not present in the vocab.
        # In that case we choose one of the factors randomly
        if not justification_v or not justification_v.has_vector:
            return random.choice(factors)

        similarities = {
            f: justification_v.similarity(self.factor_to_vec[f]) for f in factors
        }
        result = max(similarities, key=similarities.get)
        return result

    def _find_selected_factors(self, x: pd.Series) -> List[str]:
        selected_factors = []
        for factor in self.factors:
            if x.loc[factor] == 1:
                selected_factors.append(factor)
        return selected_factors

    def process_text(self, text):
        doc = self.embed(text.lower())
        result = []
        for token in doc:
            if token.text in self.embed.Defaults.stop_words:
                continue
            if token.is_punct:
                continue
            if token.lemma_ == "-PRON-":
                continue
            result.append(token.lemma_)
        return " ".join(result)


if __name__ == "__main__":
    # saves preprocessed annotations
    preprocess = Preprocess()
    preprocess.save()
