"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from pandas_datareader import wb
import pycountry_convert as pc
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import os


from explain_model_errors import CLIPModelExplanations


# data paths:
SHAPE_FILE = "/tmp/geodata/ne_10m_admin_0_countries_lakes"
# NOTE: Download https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries_lakes.shp
# and https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries_lakes.shx
# and https://github.com/nvkelso/natural-earth-vector/raw/master/10m_cultural/ne_10m_admin_0_countries_lakes.dbf
# and put these all in a folder called /tmp/geodata


# map properties:
MAP_SIZE = [24, 10]
MAP_RESOLUTION = "i"  # values: "c"rude, "l"ow, "i"ntermediate, "h"igh, "f"ull
MAP_PROJECTION = "cyl"
MAP_AREA_THRESHOLD = 20000

# colors:
ACCURACY_COLOR_MAP = "RdYlGn"
LIGHT_GRAY = (0.9, 0.9, 0.9)
GRAY = (0.4, 0.4, 0.4)

SAVE_DIR = "plots/"


def draw_world_map():
    """
    Function that draws a base world map.
    """

    # create plot of pre-specified size:
    plt.clf()
    plt.figure(figsize=MAP_SIZE)

    # draw world map:
    world_map = Basemap(
        projection=MAP_PROJECTION,
        resolution=MAP_RESOLUTION,
        area_thresh=MAP_AREA_THRESHOLD,
        lon_0=11.0,  # do not split Russia
        lat_0=90.0,
    )
    world_map.drawmapboundary(linewidth=0.0)
    world_map.drawcountries(linewidth=0.6)
    world_map.drawstates(linewidth=0.2)
    world_map.fillcontinents(color=LIGHT_GRAY, zorder=1)

    # draw only the coast lines we want:
    coasts = world_map.drawcoastlines(linewidth=0.0)
    coast_paths = coasts.get_paths()
    paths_to_keep = [idx for idx in range(64)] + [119]  # coasts but no rivers
    for path_idx in paths_to_keep:
        path = coast_paths[path_idx]
        vertices = [
            (vertex[0], vertex[1])
            for (vertex, code) in path.iter_segments(simplify=False)
        ]
        px = [vertices[idx][0] for idx in range(len(vertices))]
        py = [vertices[idx][1] for idx in range(len(vertices))]
        world_map.plot(px, py, linewidth=0.6, color="black", zorder=1)

    # return world map:
    return world_map


def correct_country_acc(accuracy_per_country):
    missed = []
    multiword_mapping = {
        "United": "United Kingdom",
        "South": "South Africa",
        "Korea,": "South Korea",
        "Ivory": "Ivory Coast",
        "Burkina": "Burkina Faso",
        "Papa": "Papua New Guinea",
        "Czech": "Czech Republic",
        "Sri": "Sri Lanka",
        "Kyrgizstan": "Kyrgyzstan",
        "USA": "United States",
        "Somaliland": "Somalia",
    }
    #''Kyrgyz Republic', 'USA': 'United States'}
    corrected = {}
    for country in accuracy_per_country.keys():
        try:
            country_name = country
            if country in multiword_mapping:
                country_name = multiword_mapping[country]
            # country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
            corrected[country_name] = accuracy_per_country[country]
        except:
            missed.append(country)
    print(missed)
    return corrected


def chloropleth_map(country_value_map, clim=None):
    """
    Creates chloropleth map of the specified values (in a country-value map).
    """

    # draw world map:
    world_map = draw_world_map()
    world_map.readshapefile(SHAPE_FILE, name="world", drawbounds=False)

    # map values to polygons based on country name:
    patches, values, countries_seen = [], [], []
    for info, shape in zip(world_map.world_info, world_map.world):

        # there are two keys that may contain correct country name:
        for country_key in ["NAME_LONG", "GEOUNIT"]:
            country = info.get(country_key, None)
            country = unidecode.unidecode(country)  # removes accented characters
            country = country.rstrip(
                "\x00"
            )  # country names contain empty characters at the end

            if country in list(country_value_map.keys()):
                polygon = Polygon(np.array(shape), True)
                patches.append(polygon)
                values.append(country_value_map[country])
                countries_seen.append(country)
                break

    # check that we did not miss any countries:
    for country in country_value_map.keys():
        if country not in countries_seen:
            print(f"Unable to plot accuracy for country: {country}")

    # create chloropleth map:
    chloropleth = PatchCollection(patches, alpha=1.0, zorder=3, cmap=ACCURACY_COLOR_MAP)
    chloropleth.set_array(np.array(values))
    if clim is not None:
        chloropleth.set_clim(clim)
    axes = plt.gca()
    axes.add_collection(chloropleth)

    # add colorbar:
    cbar = world_map.colorbar(
        mappable=chloropleth,
        location="right",
        size="2%",
        format=matplotlib.ticker.PercentFormatter(decimals=0),
    )  #'%i')
    cbar.ax.tick_params(labelsize=18)


def plot_top5_accuracy_map(model_name="clip_vit_b32", save_fig=False):
    clip_explanations = CLIPModelExplanations(model_name=model_name)
    clip_predictions = clip_explanations.annotated_predictions
    top_5_per_country_df = (
        clip_predictions.groupby("full_image_id").max().groupby("country").mean()
    )
    top_5_per_country_df = 100 * top_5_per_country_df["acc5"]
    top_5_per_country = top_5_per_country_df.to_dict()
    corrected = correct_country_acc(top_5_per_country)
    chloropleth_map(corrected)
    if save_fig:
        save_path = os.path.join(SAVE_DIR, f"{model_name}_top5_geopgraphy_country.pdf")
        plt.margins(x=0)
        plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
        params = {
            "text.usetex": True,
            "font.size": 11,
            "font.family": "lmodern",
        }
        plt.rcParams.update(params)
        plt.savefig(save_path, bbox_inches="tight")


def plot_top5_accuracy_per_region_map(model_name="clip_vit_b32", save_fig=False):
    clip_explanations = CLIPModelExplanations(model_name=model_name)
    clip_predictions = clip_explanations.annotated_predictions
    unique_imgs = clip_predictions.groupby("full_image_id").max()
    top_5_per_region_df = unique_imgs.groupby("region").mean()
    top_5_per_region_df = 100 * top_5_per_region_df["acc5"]
    imgs_with_top5_per_region = unique_imgs.join(
        top_5_per_region_df, on="region", rsuffix="_per_region"
    )
    top_5_per_region_data = (
        imgs_with_top5_per_region.groupby("country").mean()["acc5_per_region"].to_dict()
    )
    corrected = correct_country_acc(top_5_per_region_data)
    chloropleth_map(corrected)
    if save_fig:
        save_path = os.path.join(SAVE_DIR, f"{model_name}_top5_geopgraphy_region.pdf")
        plt.margins(x=0)
        plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
        params = {
            "text.usetex": True,
            "font.size": 11,
            "font.family": "lmodern",
        }
        plt.rcParams.update(params)
        plt.savefig(save_path, bbox_inches="tight")
