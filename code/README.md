# CD-APDM — code

## Modules and their status in the reported configuration

| Module | File | Phase | In the reported configuration |
|---|---|---|---|
| MDIWM (domain-aware instance weighting) | `src/modules/mdiwm.py` | Training | used; `--ablation w/o-MDIWM` removes it |
| CIWLM (class-balanced focal loss of Cui et al. 2019) | `src/modules/ciwlm.py` | Training | used; `--ablation w/o-CIWLM` removes it |
| TFEM and MHSA fusion (temporal branch) | `src/modules/tfem.py`, `src/models/backbone.py` | Both | disabled (`tfem.enabled: false`) |
| AdaBN | `src/inference.py` (`adabn`) | Inference | used (`ss_plam.adapt_mode: adabn`) |
| Post-hoc logit adjustment | `src/inference.py` | Inference | used, λ = 0.5 (`logit_adjust.tau`) |
| ACRM tail-class rectification | `src/modules/acrm.py` | Inference | used, γ = 0.005; the adaptive threshold is not used |
| CC-GANM, SS-PLAM, Mean-Teacher | `src/models/cc_gan.py`, `src/modules/ss_plam.py` | Inference | disabled; reported as negative results (Table 2) |

`configs/cd_apdm.yaml` is the reported configuration. The inference-time ablations of
Table 2 are selected with `src/inference.py --ablation`, and `ss_plam.adapt_mode` switches
between `adabn`, `mean_teacher`, `adabn_mt` and `none`.

## Running

```bash
python -m src.train --config configs/cd_apdm.yaml --seed 0            # writes cd_apdm_bestval.pt and cd_apdm_final.pt
python -m src.inference --config configs/cd_apdm.yaml --seed 0 --checkpoint <run dir>/cd_apdm_bestval.pt
```

Checkpoints are selected by Top-1 accuracy on the class-balanced source validation set;
target-domain labels are never read during training or adaptation (`StripLabels` removes
them from every loader that adaptation uses) and are used only to compute metrics.
Unlabeled target images, the test split included, are used by AdaBN and by the domain
classifier of MDIWM.

## Baselines (Table 1)

```bash
python -m src.baselines.run --method {source_only,focal,dann,wb_dann,flexmatch,swin_b} --seed 0
python -m src.baselines.run --method cotta --seed 0 --init_from <source_only run dir, same seed>
python -m tools.baseline_adabn --run_dir <baseline run dir> --ckpt bestval --out <npz>   # logits before and after AdaBN, for the AdaBN rows of Table 1
python -m pytest tests/test_baselines.py -q  # numerical checks against the official implementations in tests/official/
```

Training budget, data, augmentation and evaluation are shared with CD-APDM (read from
`configs/cd_apdm.yaml`); method-specific settings, taken from the official implementations
and not tuned on this benchmark, and every deviation from them are listed in
`configs/baselines.yaml`. Post-hoc logit adjustment of the baselines is applied offline to
the saved predictions by `scripts/analysis/aggregate_table1.py`.

## Data

Split files are read from `../dataset/splits/`; images are read from
`$AGRI_WORKSPACE/data/` (see `../scripts/fetch-datasets.sh`). If an image listed in a
split file is missing, the loader raises an error rather than substituting other input.
