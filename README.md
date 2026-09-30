# CD-APDM: a controlled laboratory-to-field test (PlantVillage-LT → PlantDoc)

Code, benchmark split files, run outputs and analysis scripts for the article
*Domain Adaptation with Long-Tail Handling for Plant Disease Recognition: A Controlled
Laboratory-to-Field Test of CD-APDM* (submitted to *Array*).

Everything needed to recompute the tables and figures of the article from the saved
predictions is included and runs on a CPU without the images. Re-training requires the
public PlantVillage and PlantDoc images, which are not redistributed here.

## Contents

| Path | What it holds |
|---|---|
| `dataset/splits/` | The benchmark: source and target split files, class mapping, duplicate groups and label-conflict flags (see `dataset/splits/README.md`) |
| `code/` | CD-APDM and the baselines of Table 1 (see `code/README.md`) |
| `scripts/analysis/` | Construction of the benchmark, the leakage audit, and every aggregation behind the tables and figures |
| `scripts/queues/` | The exact command of every run, one line per run (`<run name>` TAB `<command>`) |
| `scripts/fetch-datasets.sh` | Downloads PlantVillage and PlantDoc from their original repositories |
| `results/*.json` | Aggregated results: each table and figure of the article is generated from one of these files |
| `results/remote/` | Raw output of every run: per-image target probabilities, per-epoch training history, logs and a manifest with the code commit, environment and GPU |

## The benchmark in brief

* **Source: PlantVillage-LT.** The 28 PlantVillage classes that also occur in PlantDoc,
  subsampled exponentially with a fixed seed to a long tail (target imbalance factor 500,
  realised 487.0): 23,688 images. PlantVillage itself is close to balanced, so the long tail
  is constructed, while the laboratory-to-field shift is naturally occurring.
* **Model selection** uses a class-balanced source validation set of 1,400 images (50 per
  class): 1,200 PlantVillage images not selected by the subsampling and 200 from four head
  classes whose available images the subsampling used up. The remaining 23,488 images are
  used for training. No near-duplicate or same-leaf group spans training and validation.
* **Target: PlantDoc**, 2,566 field images of the same 28 classes (the 2,578 image files
  of the repository minus 12 files that differ from another file only in the case of their
  name), split by class and by duplicate group into a development split (782 images), used
  to choose inference-time parameters, and a test split (1,784 images), used only for the
  reported metrics.
* **Leakage audit.** Images are grouped by a 256-bit perceptual hash over the eight
  rotations and reflections of each image and by the leaf-number token that some
  PlantVillage file names carry. In a proportional 10% validation split used earlier,
  49 of 2,372 validation images shared a leaf with the training set; hashing alone finds 3
  of them. The earlier split is kept as `dataset/splits/plantvillage_lt_source_f06c00b.csv`
  for the leakage-effect experiment (`results/exp-5-leakage-effect.json`).

## Recomputing the results (CPU, no images needed)

```bash
pip install -r code/requirements.txt scipy matplotlib
python scripts/analysis/aggregate_table1.py results/remote/exp2 results/remote/exp2-ours results/remote/exp2-adabn \
       --out results/exp-2-table1.json --md results/exp-2-table1.md        # Table 1, Tables S3–S6
python scripts/analysis/aggregate_table2.py results/remote --out results/exp-3-4-table2.json --md results/exp-3-4-table2.md   # Table 2
python scripts/analysis/offline_inference_analyses.py --out results/exp-offline-inference.json   # Tables 3–4, Figs 4–5, Section 4.6
python scripts/analysis/exp9_10.py results/remote/exp9-10 --out results/exp-9-10.json            # Tables S8–S9
python scripts/analysis/exp5_leakage_effect.py results/remote/exp-final --out results/exp-5-leakage-effect.json
python scripts/analysis/make_figures.py                                                           # Figs 2–5 into figures/
```

Each command overwrites a file that is already included, so a `git diff` after running it
shows whether the result is reproduced.

## Re-running the experiments

```bash
export AGRI_WORKSPACE=$HOME/agri-cnz-workspace      # images are stored under $AGRI_WORKSPACE/data
bash scripts/fetch-datasets.sh                       # PlantVillage (raw/color) and PlantDoc
cd code
python -m src.train --config configs/cd_apdm.yaml --seed 0
python -m src.inference --config configs/cd_apdm.yaml --seed 0 --checkpoint <run dir>/cd_apdm_bestval.pt
python -m src.baselines.run --method source_only --seed 0
```

The line of `scripts/queues/<queue>.tsv` named after a run directory in
`results/remote/<queue>/` is the command that produced it, with `$PY` the Python
interpreter and `$RUN_DIR` the output directory. Every method was run with seeds 0–5 on an
NVIDIA RTX A6000. Model checkpoints are not included because of their size.

## Not included

* **Images.** Obtain PlantVillage and PlantDoc from their repositories
  (`scripts/fetch-datasets.sh`). The only images included are the eight PlantDoc test images
  shown in Fig. 2 (`results/remote/exp-final/figdata-s0/images/`).
* **The AgriNet sensor series.** The temporal module of CD-APDM is disabled in every
  reported experiment, because no public image dataset records the acquisition time and
  plot needed to pair an image with a sensor series, and no reported result depends on the
  series. The loader code is kept in `code/src/datasets/agrinet.py` but has no data to load.

## Licence

Code: MIT (see `LICENSE`). Split files, run outputs and aggregated results: CC BY 4.0.
The eight PlantDoc images are distributed under the CC BY 4.0 licence of PlantDoc
(Singh et al. 2020, *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD).
PlantVillage (Hughes and Salathé 2015) images are not redistributed.
