# MobileNetV2 Quantization

## Project Overview

This repository contains the implementation to evaluate the trade-offs in quantizing MobileNetV2's inverted residual blocks with different bit precision (2-bit, 4-bit, 8-bit) and adaptive precision quantization. The quantization configurations aim to reduce memory and computational requirements, making the model more efficient for deployment on edge devices with limited resources.

### Key Quantization Configurations:
- **Adaptive Precision**: Each block receives a variable precision depending on its computational cost. Blocks with high MACs are assigned lower bit precision (e.g., 2-bit), while blocks with fewer MACs retain higher precision (e.g., 8-bit).
- **Fixed 2-bit, 4-bit, and 8-bit**: Fixed quantization is applied uniformly to all inverted residual blocks for comparison.

### Key Benefits:
- **Optimized Computational Efficiency**: Lower bit precision significantly reduces memory and computational demands, enabling real-time processing on mobile and edge devices.

- **Trade-Off Between Accuracy and BitOps**: The adaptive quantization strategy ensures minimal performance degradation while achieving significant efficiency gains.

- **Scalable and Hardware-Friendly**: The quantized model can be easily deployed on various hardware platforms, improving scalability for real-world applications.

## Requirements

To run this project, you will need:
- **Pretrained MobileNetV2 Model**: A pretrained MobileNetV2 model to serve as the baseline.
- **Quantization Libraries**: A deep learning framework (e.g., PyTorch) to apply the quantization techniques.
- **Basic Understanding of Quantization**: Familiarity with quantization techniques and their impact on model performance.

### Installation Instructions
To run this project, you will need:

PyTorch: A deep learning framework used for loading and quantizing the MobileNetV2 model.

Pretrained MobileNetV2 Model: Extracted from the torchvision model zoo for baseline evaluation.

Matplotlib and Tabulate: For visualizing and summarizing results.

Basic Understanding of Quantization: Familiarity with quantization techniques and their impact on computational efficiency.
1. Clone the repository:


git clone https://github.com/your_username/your_repo_name.git



2. Install the required libraries using the following command:

pip install torch torchvision matplotlib



## How to Use

1. The script automatically loads a pretrained model.
2. Apply Quantization: Choose between fixed or adaptive precision quantization using command-line arguments or modifying the script.
3. Compute and Compare BitOps: The script computes the total BitOps for each configuration and visualizes the results.
4. Analyze Results: Generate plots and tables to compare the computational efficiency of different quantization strategies.

## Results

The results are shown in the form of **BitOps** comparisons across different quantization configurations. A bar chart is generated to illustrate the impact of fixed vs. adaptive quantization on computational cost. Adaptive precision significantly reduces BitOps while maintaining reasonable accuracy, making it a practical choice for deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


