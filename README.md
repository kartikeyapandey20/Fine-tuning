# Fine-tuning Llama 2 with QLoRA

This repository demonstrates fine-tuning the Llama 2 language model using QLoRA (Quantized Low-Rank Adaptation) technique. The implementation is optimized for environments with limited computational resources, such as Google Colab.

## Project Links
- Kaggle Notebook: [Fine-tuning Llama 2](https://www.kaggle.com/code/guitaristboy/fine-tuning-llama-2)

## Overview

This project shows how to fine-tune the Llama 2 7B chat model using parameter-efficient techniques. It uses 4-bit precision and QLoRA to reduce VRAM usage while maintaining model performance. The approach is particularly useful for environments with limited GPU resources.

## Requirements

Required packages:
- accelerate (v0.21.0)
- peft (v0.4.0)
- bitsandbytes (v0.40.2)
- transformers (v4.31.0)
- trl (v0.4.7)

Additional requirements:
- A Hugging Face account and API token
- Access to the Llama 2 model (requires Meta approval)
- GPU with at least 15GB VRAM

## Dataset

The project uses the Guanaco dataset reformatted to follow the Llama 2 chat template:
- Original dataset: timdettmers/openassistant-guanaco
- Reformatted dataset (1k samples): mlabonne/guanaco-llama2-1k

The dataset follows the Llama 2 chat template structure with:
- System Prompt (optional)
- User Prompt (required)
- Model Answer (required)

## Configuration

### Model Settings
- Base model: NousResearch/Llama-2-7b-chat-hf
- Output model: Llama-2-7b-chat-finetune

### QLoRA Parameters
- LoRA rank (r): 64
- LoRA alpha: 16
- LoRA dropout: 0.1

### Training Parameters
- Precision: 4-bit (NF4 quantization)
- Batch size: 4
- Learning rate: 2e-4
- Weight decay: 0.001
- Training epochs: 1
- Optimizer: AdamW (32-bit)
- Learning rate schedule: Cosine
- Warmup ratio: 0.03

## Process Overview

1. **Setup**: Configure environment and install dependencies
2. **Data Preparation**: Load and format the dataset according to Llama 2 template
3. **Model Configuration**: Set up the model with QLoRA and 4-bit quantization
4. **Training**: Fine-tune the model using the specified parameters
5. **Saving**: Save the trained model for future use
6. **Inference**: Use the model for text generation tasks

## Features

- Automatic device placement for optimal resource usage
- Gradient checkpointing for reduced memory usage
- TensorBoard integration for training monitoring
- Optimized for Google Colab environment
- Uses Llama 2 chat template format

## Limitations

- Requires at least 15GB VRAM
- Limited to 4-bit precision for memory efficiency
- Training on larger datasets may require more computational resources
- Full fine-tuning is not possible due to memory constraints
