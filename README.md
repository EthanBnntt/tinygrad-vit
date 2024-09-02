# Vision Transformer (ViT) Implementation in Tinygrad

This repository provides a minimalist implementation of the Vision Transformer (ViT) model using `tinygrad`. The Vision Transformer applies transformer architecture to image classification tasks by treating image patches as sequences, similar to words in natural language processing.

## Overview

The Vision Transformer (ViT) model was introduced by Google Research and represents a novel approach in computer vision. Unlike traditional convolutional neural networks (CNNs), which process images through convolutional layers, ViT divides an image into fixed-size patches and processes them as sequences using transformer mechanisms. This approach leverages the power of transformers, originally designed for NLP tasks, to handle image data.

## Features

- Minimalist implementation of the ViT model.
- Utilizes the `tinygrad` framework for training and inference.
- Supports image classification tasks with customizable model parameters.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/EthanBnntt/tinygrad-vit.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd tinygrad-vit
   ```

3. **Install the Required Dependencies:**

   Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the Vision Transformer model on the MNIST dataset, execute the following command:

```bash
python train.py
```

This script will:

- Load the MNIST dataset.
- Initialize and train the ViT model.
- Output training loss and accuracy metrics.

## Configuration

You can adjust various model parameters in the `train.py` script:

- **Image Dimensions**: Set `image_width` and `image_height` based on your dataset.
- **Patch Size**: Modify `patch_width` and `patch_height` to match the image patch size.
- **Model Hyperparameters**: Adjust `embed_dim`, `hidden_dim`, `num_heads`, `dropout_p`, `num_layers`, and `mlp_gating` to optimize model performance.

## Example

Here’s a brief example of how to use the `ViTModel` class:

```python
from model import ViTModel

# Initialize the model
model = ViTModel(
    image_width=28,
    image_height=28,
    patch_width=7,
    patch_height=7,
    channels=1,
    embed_dim=256,
    hidden_dim=512,
    num_heads=32,
    dropout_p=0.15,
    bias=True,
    num_layers=2,
    mlp_gating=True,
)

# Use the model
# model(x) where x is your input tensor
```

## Contributing

Feel free to submit pull requests and report issues. Contributions to improve the implementation, add features, or fix bugs are welcome!

