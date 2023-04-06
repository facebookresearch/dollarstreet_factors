"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from annotations import Annotations
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests
from io import BytesIO
import plotly.io as pio
from PIL import Image
import os


class PlotExamplesAcrossIncomes:
    def __init__(
        self,
        data_dir: str = "data/",
        save_dir: str = "plots/",
    ):
        labels = Annotations(data_dir=data_dir)
        self.table = labels.annotations
        self.save_dir = save_dir

    def show_images_across_incomes(
        self, class_label="roofs", num_images: int = 5
    ) -> go.Figure:

        high_income_urls = self.table[
            (self.table["class"] == class_label)
            & (self.table["income_bucket"] == "low")
        ]["url"].values[:num_images]

        low_income_urls = self.table[
            (self.table["class"] == class_label)
            & (self.table["income_bucket"] == "high")
        ]["url"].values[:num_images]

        fig = make_subplots(
            rows=2,
            cols=num_images,
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )

        for i, url in enumerate(low_income_urls):
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            fig.add_trace(go.Image(z=img), 1, i + 1)

        for i, url in enumerate(high_income_urls):
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            fig.add_trace(go.Image(z=img), 2, i + 1)

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            title_text=f"{class_label} high (top) versus low (bottom) incomes"
        )
        return fig

    def save(self):
        class_label = "roofs"
        fig = self.show_images_across_incomes(class_label=class_label)
        pio.full_figure_for_development(fig, warn=False)
        save_path = os.path.join(self.save_dir, f"{class_label}_images.pdf")
        fig.write_image(save_path)
        fig.write_image(save_path, scale=5.0)

        class_label = "ceilings"
        fig = self.show_images_across_incomes(class_label=class_label)
        pio.full_figure_for_development(fig, warn=False)
        save_path = os.path.join(self.save_dir, f"{class_label}_images.pdf")
        fig.write_image(save_path, scale=5.0)


if __name__ == "__main__":
    plots = PlotExamplesAcrossIncomes()
    plots.save()
