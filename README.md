# Fashion Item Classification using MLP — Neural Networks Project

## Problem Description
This project implements a Multilayer Perceptron (MLP) to classify fashion items from the Fashion-MNIST dataset into 10 categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Dataset
- **Name:** Fashion-MNIST
- **Source:** https://github.com/zalandoresearch/fashion-mnist
- **Loaded via:** `torchvision.datasets.FashionMNIST`
- **Size:** 70,000 grayscale images (28×28 pixels)
- **Split:** 50,000 train | 10,000 validation | 10,000 test

## Model Architecture
A fully connected MLP with the following structure:

```
Input (784) → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 10)
```

## Preprocessing
- **Missing values:** None — Fashion-MNIST is a pre-cleaned benchmark dataset
- **Normalization:** `transforms.Normalize(mean=0.5, std=0.5)` — rescales pixels from [0,1] to [-1,1]
- **Splitting:** 50,000 train / 10,000 validation / 10,000 test using `random_split`

## Training
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 20 max with early stopping (patience=5)
- **Batch size:** 64

## Results

### Experiment Comparison

| Setting              | Experiment 1 (ReLU) | Experiment 2 (Tanh) |
|----------------------|---------------------|---------------------|
| Activation function  | ReLU                | Tanh                |
| Learning rate        | 0.001               | 0.001               |
| Architecture         | 784→256→128→64→10   | 784→256→128→64→10   |
| Final train accuracy | 93.13%              | 91.36%              |
| Test accuracy        | 88.46%              | 87.51%              |
| Final loss           | 0.3796              | 0.3621              |
| Epochs until stop    | 16                  | 17                  |

### Conclusion
- ReLU outperformed Tanh in test accuracy (88.46% vs 87.51%)
- Tanh achieved slightly lower final loss (0.3621 vs 0.3796)
- Both models struggled most with the "Shirt" class (F1: 0.70 and 0.66) due to its visual similarity with T-shirts, Pullovers, and Coats
- ReLU is the better model overall for this task

## How to Run
1. Open the notebook in Google Colab
2. Run all cells in order from top to bottom
3. The dataset will download automatically on first run

## Requirements
```
torch
torchvision
matplotlib
numpy
scikit-learn
```
