# MobileNetV2 Quantization

## Project Overview

This repository contains the implementation to evaluate the trade-offs in quantizing MobileNetV2's convolutional layers with different bit precision (2-bit, 4-bit, and 8-bit). The goal of this project is to optimize the model's performance and computational efficiency, particularly for deployment on edge devices and resource-constrained hardware.

## Task Description

This task investigates the impact of quantization on the MobileNetV2 architecture by adjusting the precision of its convolutional layers. The precision reductions lead to significant reductions in memory and computational requirements, making the model more efficient for deployment without compromising too much on performance.

### Key Benefits:
- **Efficient Resource Utilization**: Lower bit precision helps reduce memory and processing power required for deploying models on mobile and edge devices.
- **Model Performance Optimization**: Balancing computational efficiency and accuracy, especially in resource-limited settings.
- **Scalability**: Enabling deployment of real-time applications such as image recognition.

## Requirements

To run this project, you will need:
- **MobileNetV2 Model**: A pretrained MobileNetV2 model used as the baseline.
- **Quantization Libraries**: Libraries from a deep learning framework (like PyTorch) to apply quantization techniques.
- **Computational Resources**: A development environment that can run deep learning models with at least CPU processing. Access to a GPU can speed up the training and evaluation process.
- **Basic Knowledge**: Familiarity with quantization techniques and how they affect model performance.

### Installation Instructions

1. Clone the repository:

git clone https://github.com/your_username/your_repo_name.git



2. Install the required libraries using the following command:

pip install torch torchvision matplotlib



## How to Use

1. Load the pretrained MobileNetV2 model.
2. Apply quantization (2-bit, 4-bit, or 8-bit) using the provided scripts.
3. Evaluate the performance and resource utilization using BitOps metrics.

## Results

The results are shown in the form of **BitOps** comparisons across different quantization configurations. You can visualize these results using the provided plots.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


