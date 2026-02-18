# pytorch-image-classification
Deep learning image classification project using PyTorch (CNN on CIFAR-10 dataset).
# PyTorch Image Classification

This repository contains a Convolutional Neural Network (CNN) implemented in PyTorch for image classification on the CIFAR-10 dataset.

## Project Structure
pytorch-image-classification/ │ ├── data/                       ← CIFAR-10 dataset ├── models/ │   └── cnn.py                  ← CNN model ├── train.py                     ← Training script ├── README.md                    ← Project documentation └── requirements.txt             ← Python dependencies
## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
How to run
Make sure you have Python and PyTorch installed.
Run the training script:
python train.py
The model will train for 5 epochs on CIFAR-10 and print the loss every 100 mini-batches.
Author
Mohammadreza Mohammadi
