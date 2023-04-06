# Dollar Street Factor Annotations

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


[Paper](https://dollarstreetfactors.metademolab.com/) | [Dashboard](https://dollarstreetfactors.metademolab.com/)

<img width="1710" alt="image" src="https://user-images.githubusercontent.com/6558776/229595599-fdb951b4-fd67-4075-92d3-665980400f63.png">


## Loading Dollar Street Factor Annotations

Make sure pandas is installed (`pip install pandas`). Then load annotations table as:

```python
import pandas as pd

df = pd.read_csv("data/DollarStreetFactors.csv", index_col=0)
```

![image](https://user-images.githubusercontent.com/6558776/229595381-12ac9541-d4f7-40b8-8c44-a1eaf5128cc8.png)


<details>
  <summary>Definitions of fields</summary>
  
  Each row represents a Dollar Street image:
  - factors such as pose, lighting etc. are indicated with a bool (1: meaning the factor was selected as distinctive)
  - one_word: refers to the free-form text descriptions of distinctive factors annotators provided (in one word summaries)
  - justification: refers to the free-form question asking annotators to explain why they selected the set of factors as distinctive
  - agree right: indicates whether the annotators agrees with the label for the image.
  - why disagree: asks for an explanation of why the annotators disagreed (if agree right was marked as false).
  
</details>


# DollarStreet Data

We use the version of DollarStreet from ["Fairness Indicators for Systematic Assessments of Visual Feature Extractors"](https://github.com/facebookresearch/vissl/blob/main/projects/fairness_indicators/geographical_diversity_indicator2.md).

Please see LICENSE file for usage restrictions.
