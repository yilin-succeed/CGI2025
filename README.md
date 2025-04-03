### AMNet: Attention-Enhanced Multi-Branch Network for Micro-Expression Recognition
## Submission to CGI 2025 (The Visual Computer Journal)

This repository contains the implementation of AMNet, a novel attention-enhanced multi-branch network designed for micro-expression recognition (MER). AMNet addresses the challenges of subtle, short-duration micro-expressions by integrating advanced attention mechanisms, spatiotemporal modeling, and hierarchical feature fusion. The code is developed to support experiments on benchmark datasets such as CASME II, SAMM, SMIC, and CAS(ME)³, achieving state-of-the-art performance as detailed in our paper.
The final code will be fully released to the public upon acceptance of the paper at CGI 2025.

## Repository Structure
The repository includes the following files:
CA_block.py: Implements the Improved Multi-Modal Attention (IMMA) mechanism, combining channel, spatial, and diagonal attention to enhance feature extraction.
data.py: Handles data preprocessing, including face alignment, resizing, and augmentation for consistent input across datasets.
dataset.py: Defines the dataset loading logic (e.g., SAMMDataset) for fetching micro-expression sequences and optical flow features.
model.py: Defines the AMNet architecture, integrating optical flow, motion, and spatiotemporal branches with hierarchical fusion.
magnet.py: Implements an improved MagNet model for motion representation extraction, enhancing dynamic feature capture.
train.py: Contains the training script, including loss function (weighted cross-entropy), optimizer (AdamW), and evaluation metrics (UF1, UAR).

## Prerequisites
Python 3.8+
PyTorch 2.0.0+ with CUDA 11.8 (for GPU support)
NVIDIA GPU (e.g., RTX 3090 recommended)
