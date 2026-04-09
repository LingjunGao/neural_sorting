# NEURAL SORTING FOR FAST 3D GAUSSIAN SPLATTING

Neural Sorting is a 3D Gaussian Splatting variant integrated into this repository for efficient training and evaluation on mip-NeRF360 benchmark scenes.

## Installation

### Install from source

Clone and enter the repository:

```bash
git clone --recurse-submodules <your-repo-url>
cd neural_sorting
```

### One-command setup (recommended)

From the repository root, run:

```bash
source ./setup.sh
```

This command will:
- install required system packages,
- create and activate the `neural_sorting` conda environment,
- install Neural Sorting and Python dependencies,
- download mip-NeRF360 data,
- download benchmark checkpoints from Zenodo.

> If you run `bash ./setup.sh`, setup still works, but conda activation will not persist in your current shell.

## Usage

### Training (optional)

Training is optional because `setup.sh` already downloads benchmark checkpoints.

To fine-tune / train from those checkpoints, run:

```bash
cd examples
bash benchmarks/Train_mlp.sh
```

This script writes outputs to:
- `examples/results/mlp_checkpoint_new/`

### Evaluation

To evaluate, run:

```bash
cd examples
bash benchmarks/Eval_mlp.sh
```

Checkpoint selection order in `Eval_mlp.sh`:
1. `examples/results/mlp_checkpoint_new/` (if you ran `Train_mlp.sh`)
2. `examples/results/mlp_checkpoint/` (downloaded during setup)

## Acknowledgements

This work builds on the [gsplat library](https://github.com/nerfstudio-project/gsplat) developed by the Nerfstudio team.

If you find gsplat useful in your projects or papers, please consider citing:

```
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting},
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765},
}
```
