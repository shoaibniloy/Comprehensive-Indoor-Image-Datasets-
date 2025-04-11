
# Comprehensive Indoor Image Datasets

This repository contains a comprehensive collection of indoor image datasets, designed for training object detection models with YOLOv8. The datasets are categorized into **training**, **validation**, and **test** sets, and labeled according to a wide range of indoor objects, including furniture, appliances, tools, and animals. The dataset includes 184 different object classes.

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Classes](#classes)
- [Training the Model](#training-the-model)
- [Resuming Training](#resuming-training)
- [Downloading Model Checkpoints](#downloading-model-checkpoints)
- [License](#license)

## Overview

The dataset is organized for training an object detection model using YOLOv8. It contains images and labels that correspond to 184 indoor object classes. This repository also includes a YAML file configuration for YOLOv8, allowing you to train the model on your custom dataset with ease.

### Key Features:
- 184 object classes for diverse indoor environments.
- Split into **train**, **valid**, and **test** directories.
- YOLOv8-compatible annotation format.
- Pre-configured YAML file for YOLOv8 training.
- Support for resuming training from model checkpoints.

## Dataset Structure

The dataset is structured in the following way:

```
/content/dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

- **images/**: Contains the images for training, validation, and testing.
- **labels/**: Contains the YOLO annotations corresponding to the images.

## Classes

The dataset includes 184 object classes. Some of the object classes are:
- **gun**
- **Aggressor**
- **Person**
- **Plants**
- **Power-Tool**
- **Printer**
- **Shelve-Storage**
- **Shoe-Rack**
- **Side-Table**
- **Standard-Sofa**
- **TV-Stand**
- **cat-Abyssinian**
- **dog-american_bulldog**
- **Washing-Machine**
- **Fridge**
- **Bed**

For the full list of object classes, please refer to the `data.yaml` file in the repository.

## Training the Model

To train the YOLOv8 model on your custom dataset, you can use the following Python code:

### 1. **Install Dependencies**

Make sure you have the required libraries installed:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
```

### 2. **Train YOLOv8 Model**

You can train the YOLOv8 model by running the following script:

```python
from ultralytics import YOLO

# Load the YOLOv8 model (using a pre-trained YOLOv8 model)
model = YOLO('yolov8n.pt')  # You can choose other YOLOv8 models if needed

# Start training the model
model.train(
    data='/content/dataset/All/data.yaml',  # Path to the data.yaml file
    epochs=100,                             # Number of training epochs
    batch=16,                               # Batch size
    imgsz=640,                              # Image size
    save=True,                              # Save the model after training
    save_period=10,                         # Save the model every 10 epochs
    project='/content/ultralytics/runs/train',  # Path to save the training run
    name='custom_yolov8',                   # Name for the training run
    pretrained=True,                        # Use pre-trained weights
)
```

This command will start the training process, using a **pre-trained YOLOv8 model** and saving checkpoints every 10 epochs.

## Resuming Training

If your training session is interrupted and you want to **resume training** from the last checkpoint, use the following code:

```python
from ultralytics import YOLO

# Load the model from the last saved checkpoint (for example, epoch_30.pt)
model = YOLO('/content/ultralytics/runs/train/custom_yolov8/weights/epoch_30.pt')

# Resume training from epoch 30
model.train(
    data='/content/dataset/All/data.yaml',  # Path to the data.yaml file
    epochs=100,                             # Total epochs to train (resume from epoch 30)
    batch=16,                               # Batch size
    imgsz=640,                              # Image size
    save=True,                              # Save the model after training
    save_period=10,                         # Save the model every 10 epochs
    project='/content/ultralytics/runs/train',  # Path to save the training run
    name='custom_yolov8',                   # Name for the training run
    pretrained=False,                       # Continue from the last checkpoint
)
```

The model will resume training from the checkpoint (e.g., `epoch_30.pt`) and continue until the desired number of epochs is completed.

## Downloading Model Checkpoints

The model will automatically save checkpoints during training. You can download the latest checkpoint (e.g., `epoch_30.pt`) as follows:

```python
import shutil

# Path to the model checkpoint
checkpoint_path = '/content/ultralytics/runs/train/custom_yolov8/weights/epoch_30.pt'

# Move the checkpoint to a download-friendly location
shutil.move(checkpoint_path, '/content/epoch_30.pt')
```

Afterward, you can download the file using the Colab interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Conclusion

This repository provides a detailed dataset and configuration to train YOLOv8 models for indoor object detection. It supports resuming training from saved checkpoints, enabling users to continue training even after interruptions. Feel free to fork, use, and modify this dataset for your own custom object detection tasks.

---
