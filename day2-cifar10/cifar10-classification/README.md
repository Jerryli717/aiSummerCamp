# CIFAR-10 Image Classification

This project implements a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Project Structure

```
cifar10-classification
├── src
│   ├── cifar10_code_to_start_with.py  # Initial setup for CIFAR image classification
│   └── models
│       └── cnn.py                      # CNN architecture definition
├── data                                 # Folder containing the CIFAR dataset
├── requirements.txt                     # List of dependencies
└── README.md                            # Project documentation
```

## Installation

To set up the project, you need to install the required dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: The CIFAR-10 dataset will be automatically downloaded and loaded when you run the script in `src/cifar10_code_to_start_with.py`.

2. **Model Training**: The script includes placeholders for creating and training the CNN model. You can modify the code to implement the training loop and evaluation metrics.

3. **CNN Architecture**: The CNN architecture is defined in `src/models/cnn.py`. You can customize the model as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.