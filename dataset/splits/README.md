# Benchmark split files

Paths are relative to the dataset roots created by `../../scripts/fetch-datasets.sh`
(`$AGRI_WORKSPACE/data/plantvillage/` and `$AGRI_WORKSPACE/data/plantdoc/`).

## `plantvillage_lt_source.csv` — source domain (24,888 rows)

| Column | Meaning |
|---|---|
| `path` | Image path under the PlantVillage repository (`raw/color/...`) |
| `class`, `class_idx` | PlantVillage class and its index in the shared 28-class label space |
| `bucket` | `many` (> 100 images), `medium` (20–100) or `few` (< 20), on the constructed long-tailed distribution (Liu et al. 2019); the tail is `medium` ∪ `few` |
| `split` | `train` (23,488) or `val` (1,400, 50 per class) |
| `leaf_group` | Duplicate group: images joined by perceptual-hash near-duplication or by a shared leaf-number token; empty if the image is in no group |
| `leak_flags` | `exact_dup`, `near_dup`, `same_leaf` (how the image entered its group) |

The long-tailed set PlantVillage-LT is the 23,488 training images plus 200 of the
validation images (from four head classes whose available images the subsampling used
up); the other 1,200 validation images are PlantVillage images not selected by the
subsampling. No `leaf_group` contains both a training and a validation image.

`plantvillage_lt_source_f06c00b.csv` is the earlier proportional split (10% of
PlantVillage-LT held out for validation, 23,688 rows), in which 49 of the 2,372
validation images shared a leaf with a training image. It is kept only for the
leakage-effect experiment (`results/exp-5-leakage-effect.json`) and, together with
`plantdoc_target_f06c00b.csv` (the target list of the same version, before the
development/test split), for recomputing the leakage of that split
(`scripts/analysis/exp5_original_split.py`).

## `plantdoc_target.csv` — target domain (2,566 rows)

| Column | Meaning |
|---|---|
| `path` | Image path under the PlantDoc repository (`train/...` or `test/...`) |
| `plantdoc_class`, `plantvillage_class`, `class_idx` | Class mapping: a PlantDoc class is mapped only when host crop and pathology both agree |
| `native_split` | PlantDoc's own train/test partition (not used) |
| `dup_group` | Near-duplicate group within PlantDoc; empty if none |
| `leak_flags` | `exact_dup`, `near_dup`, and `label_conflict` when duplicates carry different labels (66 images) |
| `eval_split` | `dev` (782; for choosing inference-time parameters) or `test` (1,784; for reported metrics) |

The development and test splits are stratified by class and share no `dup_group`.
Target labels are used only to choose inference-time parameters (development split) and to
compute metrics (test split). The class *Tomato two-spotted spider mite* has two target
images and is excluded from class-averaged metrics. The class mapping, with the classes of
each dataset that have no counterpart, is also recorded in `results/exp-1a-class-stats.json`.
