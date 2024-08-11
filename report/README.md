# Training Script for DiT using DeepSpeed

## Overview
This guide provides a comprehensive walkthrough for training a DiT (Diffusion in Transformer) model using DeepSpeed. The script is designed to leverage distributed data parallel (DDP) training across multiple GPUs.

## Software Requirements
- Python 3.7+
- PyTorch 1.8+
- DeepSpeed
- torchvision
- numpy
- diffusers

## Installation Instructions
1. **Install PyTorch**:
   ```bash
   pip install torch torchvision
   ```

2. **Install DeepSpeed**:
   ```bash
   pip install deepspeed
   ```

3. **Install diffusers**:
   ```bash
   pip install diffusers
   ```

4. **Install other dependencies**:
   ```bash
   pip install numpy
   ```

## Hardware Requirements
- At least one NVIDIA GPU (recommended: A100 for faster training)
- Sufficient VRAM (depending on the model size and batch size)

## Needed Downloads
- **Pre-trained VAE model**: Download the pre-trained VAE model from the specified path or use a default path provided in the script.
- **Dataset**: Ensure the dataset is available at the specified `data-path`.

## Configuration
Before running the script, ensure you have the following configurations set:
- **Data Path**: Specify the path to your dataset.
- **Results Directory**: Directory where results and checkpoints will be saved.
- **Model Configuration**: Choose the DiT model configuration from the available options.
- **VAE Path**: Path to the pre-trained VAE model.
- **Image Size**: Image size for training (256 or 512).
- **Number of Classes**: Number of classes in your dataset.
- **Epochs**: Number of training epochs.
- **Batch Size**: Batch size for training.
- **Global Seed**: Seed for reproducibility.
- **Number of Workers**: Number of worker threads for data loading.
- **Logging Frequency**: Frequency of logging training metrics.
- **Checkpoint Frequency**: Frequency of saving model checkpoints.
- **Local Rank**: Local rank for distributed training.

## Running the Script
To run the script, use the following command:
```bash
deepspeed --hostfile <hostfile> --include="localhost:0,1" train.py --data-path <data-path> --results-dir <results-dir> --model <model> --vae-path <vae-path> --image-size <image-size> --num-classes <num-classes> --epochs <epochs> --train_batch_size <train_batch_size> --global-seed <global-seed> --num-workers <num-workers> --log-every <log-every> --ckpt-every <ckpt-every> --local-rank <local-rank>
```

Replace the placeholders with your specific configurations.

## Script Breakdown
- **Imports**: Import necessary libraries and modules.
- **Helper Functions**: Functions for setting gradients, cleaning up DDP, creating a logger, and center cropping images.
- **Training Loop**: Main training loop that initializes the model, dataset, and optimizer, and performs the training steps.
- **Arguments Parsing**: Parses command-line arguments for configuration.
- **Main Function**: Orchestrates the training process, including model initialization, data loading, and training loop.

## Notes
- Ensure CUDA and cuDNN are properly installed and configured for optimal performance.
- Monitor GPU memory usage and adjust batch size if necessary.
- For distributed training across multiple nodes, ensure proper network configuration and use the appropriate hostfile.

By following this guide, you should be able to set up and train a DiT model using DeepSpeed efficiently.
