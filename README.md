Comprehensive Indoor Image Datasets
This repository contains a comprehensive collection of indoor image datasets, designed for training object detection models using YOLOv8. The datasets are categorized into training, validation, and test sets and labeled with a wide variety of indoor objects, including furniture, appliances, tools, and animals. The dataset includes 184 different object classes, making it ideal for training and evaluating object detection models.

Table of Contents
Overview

Dataset Structure

Classes

Training the Model

Resuming Training

Downloading Model Checkpoints

License

Overview
The dataset is structured for training an object detection model with YOLOv8. It contains images and corresponding annotations for 184 indoor object classes, such as furniture, electronics, and various household items. The repository also includes a pre-configured YAML file for YOLOv8, making it easy to integrate the dataset into a training pipeline.

Key Features:
184 object classes: A diverse set of indoor objects, including guns, plants, furniture, appliances, animals, and more.

Organized dataset: Split into train, valid, and test directories.

YOLOv8-compatible annotations: The dataset is ready to be used with YOLOv8 for object detection tasks.

Pre-configured YAML: Easily set up for training with YOLOv8.

Dataset Structure
The dataset is organized as follows:

bash
Copy
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
images/: Contains the images for training, validation, and testing.

labels/: Contains the YOLO annotations corresponding to the images.

Classes
This dataset includes 184 object classes. Some of the object classes are:

gun

Aggressor

Person

Plants

Power-Tool

Printer

Shelve-Storage

Shoe-Rack

Side-Table

Standard-Sofa

TV-Stand

cat-Abyssinian

dog-american_bulldog

Washing-Machine

Fridge

Bed

Knife

Toilet

For the complete list of object classes, refer to the data.yaml file in the repository.

Training the Model
To train YOLOv8 on your custom dataset, follow these steps:

1. Install Dependencies
First, install the required libraries:

bash
Copy
pip install torch torchvision torchaudio
pip install ultralytics
2. Train YOLOv8 Model
Run the following Python code to train the YOLOv8 model on your custom dataset:

python
Copy
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
This will start training the model, using the pre-trained YOLOv8 model and saving checkpoints every 10 epochs.

Resuming Training
If your training session is interrupted and you want to resume training from the last saved checkpoint, use the following code:

python
Copy
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
The model will resume training from the checkpoint (e.g., epoch_30.pt) and continue until the desired number of epochs is completed.

Downloading Model Checkpoints
Model checkpoints will be saved during training. You can download the latest checkpoint (e.g., epoch_30.pt) by running the following code:

python
Copy
import shutil

# Path to the model checkpoint
checkpoint_path = '/content/ultralytics/runs/train/custom_yolov8/weights/epoch_30.pt'

# Move the checkpoint to a download-friendly location
shutil.move(checkpoint_path, '/content/epoch_30.pt')
After running the above code, you can download the .pt file using the Colab interface.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Conclusion
This repository provides a detailed dataset and configuration to train YOLOv8 models for indoor object detection. It supports resuming training from saved checkpoints, enabling users to continue training even after interruptions. Feel free to fork, use, and modify this dataset for your own custom object detection tasks.
