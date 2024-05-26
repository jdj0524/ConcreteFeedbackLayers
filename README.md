# Concrete Feedback Layers

This repository contains the implementation of Concrete Feedback Layers and the Feedback Bit Masking Unit (FBMU) for variable-length, bit-level CSI feedback optimization in FDD Wireless Communication Systems.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
  - [Sample Configuration for O1_140 Scenario](#sample-configuration-for-o1_140-scenario)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Checkpoints](#checkpoints)
- [License](#license)

## Overview

The Concrete Feedback Layers facilitate efficient CSI feedback by leveraging the Gumbel-softmax technique for continuous relaxation of binary feedback. The FBMU enables true variable-length bit-level feedback. Together, these components streamline the feedback process and optimize performance across various feedback lengths.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ConcreteFeedbackLayers.git
cd ConcreteFeedbackLayers
pip install -r requirements.txt
```

## Usage

To run the main script, use the following command:

```bash
python main.py --config={config_name} --gpu={gpu_number}
```

## Configuration

All configurations for the experiments are housed in the `experiment_config.py` file located at `pytorch/configs/`. Below is a sample configuration for the O1_140 scenario.

### Sample Configuration for O1_140 Scenario

```python
def channeltransformer_sgdr_bit_variable():
    launcher = BaseLauncher
    model = ChannelTransformerBitLevelLargeVariable
    trainer = SGDR_Bit_Variable_Trainer
    data = DeepMIMOSampleDataset
    options = {
        'wandb_project_name': 'BitChannelTransformer',
        'save_dir': '/home/automatic/projects/ConcreteFeedbackLayers/checkpoints/O1_140/',
        'batch_size': 256,
        'data_options': {
            'files_dir':'/home/automatic/DeepMIMO_Datasets/O1_140_8b84b4/'
        },
        'data_split': {
            'train': 0.6, 
            'val': 0.2, 
            'test': 0.2
        },
        'model_options': {
            'n_blocks':3, 'd_model':512, 'nhead':8, 'dim_feedforward':2048, 
            'n_tx':64, 'n_rx':16, 'n_carrier':128, 'dim_feedback':256
        },
        'temp_schedule': {
            'start_temp': 10,
            'final_temp': 0.01
        },
        'trainer_options': {
            'max_bits' : 256,
            'eval_bits' : 128,
            'epochs' : 500, 
            'loss' : MSE_loss,
            'optimizer_cls' : torch.optim.AdamW,
            'gpu' : 0, 
            'metrics' : {
                'cosine' : (Cosine_distance, 'max'),
                'NMSE' : (NMSE_loss, 'min'),
            },
        },
        'optimizer_options': {
            'lr' : 1e-5,
        },
        'train_schedulers': CosineAnnealingWarmUpRestarts,
        'train_scheduler_options': 
            {
                'T_0' : 500,
                'T_mult' : 2,
                'T_up': 5,
                'eta_max': 0.001,
                'last_epoch':-1,
                'gamma': 0.8
            },
    }
    return [(launcher, model, trainer, data, options)]
```

## Running Experiments

To run experiments, specify the desired configuration name and GPU number in the command:

```bash
python main.py --config=channeltransformer_sgdr_bit_variable --gpu=0
```

Make sure the paths in the configuration file are correctly set to your dataset and checkpoint directories.

## Results

After running the experiments, checkpoints will be saved in the directory specified in the configuration file (`save_dir`).
Metrics during training and testing will be saved to WandB, in the project name provided in the configuration (`wandb_project_name`).

## Checkpoints

Pre-trained checkpoints for the O1_140 and O1_28B scenarios are provided in the `checkpoints/` directory at the project root. These checkpoints have been saved using `git-lfs` to manage large files. You can find the checkpoints in their respective subdirectories:

- `checkpoints/O1_140/`
- `checkpoints/O1_28B/`

To use these checkpoints, ensure you have `git-lfs` installed and initialized in your local repository. For more information on `git-lfs`, visit [Git LFS](https://git-lfs.github.com/).

```bash
git lfs install
git lfs pull
```

These checkpoints can be loaded and used to continue training or for evaluation purposes as specified in the configuration files.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
