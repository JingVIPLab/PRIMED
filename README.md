# [ACMMM 2026] Adaptive Modality Suppression for Referring Audio-Visual Segmentation via Biased Competition

## Installation

Use Python 3.10 and CUDA-enabled PyTorch. Run all commands from the repository root.

```bash
conda create -n primed python=3.10 -y
conda activate primed

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

CLAP is bundled in `CLAP/` and does not require a separate `msclap` installation.

## Preparation

### Pretrained models

Place the [pretrained assets](https://drive.google.com/drive/folders/1HvlIvQPYe5b3pD8jVuuHYmvRDD06XPn9?usp=sharing) in the following directory:

```text
pretrained/
├── sam2_hiera_large.pt
├── CLAP_weights_2023.pth
└── distilroberta-base
```

### Datasets

Set `path_root` and the `DATA.*` paths in the configuration for your dataset. AVSBench is also configurable.

| Dataset | CLI value | Configuration | Frames per video |
| --- | --- | --- | --- |
| Ref-AVS | `refavs` | [utils/config_refavs.py](utils/config_refavs.py) | 10 |
| AVSBench S4 | `s4` | [utils/config_s4.py](utils/config_s4.py) | 1 for training; 5 for evaluation |
| AVSBench MS3 | `ms3` | [utils/config_m3.py](utils/config_m3.py) | 5 |

Ref-AVS expects the following layout:

```text
Ref-AVS/
├── metadata.csv
├── media
└── gt_mask
```

### Checkpoint

The checkpoint is available [here](https://drive.google.com/file/d/1oCbGICzLIqPUESIy5uRoeZRS26xZPmlE/view?usp=sharing).

## Modality annotation and audit

We have provided the annotated results [here](https://drive.google.com/file/d/1xHU5qB6h2CtnNZW1JJtx-iasJgueh2-s/view?usp=sharing). You may download and replace the original metadata.csv in Ref-AVS dataset. If you would like to annotate and audit from scratch, please follow the instructions below.

The scripts in [qwen/](qwen/) use Qwen3-Omni-Flash to annotate modality-prior labels from referring expressions and audit them using the prompts from the paper. Both stages use text only.

### Setup

Prepare for your DashScope API key.

```dotenv
DASHSCOPE_API_KEY=your-api-key
```

### Annotation

The input CSV must contain `exp`. Replace the dataset path below with your local path:

```bash
python qwen/qwen_api.py \
  --input /path/to/Ref-AVS/metadata.csv \
  --output ./annotations/metadata_annotated.csv
```

The script preserves the original columns and adds `api_result` in the format `{Modality}. (Explanation: {Reasoning})`, where the modality is `Visual`, `Audio`, or `Audio-Visual`.

### Audit

```bash
python qwen/qwen_audit.py \
  --input ./annotations/metadata_annotated.csv \
  --output ./annotations/metadata_audited.csv
```

Audit responses:

```text
Correct. (Explanation: ...)
Incorrect. Correct Modality: Visual. (Explanation: ...)
```

## Training

Train on Ref-AVS with the provided script:

```bash
bash train.sh
```

## Evaluation

Evaluate with the provided script:

```bash
bash test.sh
```

## Citation

If you find this work helpful, please consider citing our work:

```text
@article{he2026primed,
  title={PRIMED: Adaptive Modality Suppression for Referring Audio-Visual Segmentation via Biased Competition},
  author={He, Yuchen and Zhang, Jing},
  journal={arXiv preprint arXiv:2605.07154},
  year={2026}
}
```

