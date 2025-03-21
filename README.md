# MobileNetV2 Quantization

## Project Overview

This repository contains the implementation to evaluate the trade-offs in quantizing MobileNetV2's convolutional layers with different bit precision (2-bit, 4-bit, 8-bit) and adaptive precision quantization. The quantization configurations aim to reduce memory and computational requirements, making the model more efficient for deployment on edge devices with limited resources.

### Key Quantization Configurations:
- **Adaptive Precision**: Layers receive a variable precision depending on their position in the network (higher precision for early layers and lower precision for deeper layers).
- **Fixed 2-bit, 4-bit, and 8-bit**: Fixed quantization applied to each convolutional layer for comparisons.

### Key Benefits:
- **Efficient Resource Utilization**: Lower bit precision reduces memory and computational requirements, which is crucial for running models on mobile and edge devices.
- **Model Performance Optimization**: Achieve a balance between computational efficiency and accuracy, adapting to specific deployment needs.
- **Scalability**: Quantization helps make models more efficient and scalable for real-world applications like real-time image recognition.

## Requirements

To run this project, you will need:
- **MobileNetV2 Model**: A pretrained MobileNetV2 model to serve as the baseline.
- **Quantization Libraries**: A deep learning framework (e.g., PyTorch) to apply the quantization techniques.
- **Computational Resources**: A development environment capable of running deep learning models with at least CPU processing. GPUs may accelerate training and evaluation.
- **Basic Understanding of Quantization**: Familiarity with quantization techniques and their impact on model performance.

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


