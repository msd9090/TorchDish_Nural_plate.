## 🍽️ DishClassifier:
Dirty vs. CleanAn automated Computer Vision solution to distinguish between dirty and cleaned dishes. This project leverages Deep Learning to help automate kitchen workflows or smart appliance monitoring.
## 🚀 OverviewThis
repository contains a PyTorch implementation for image classification on the Platesv2 Dataset. It achieves high accuracy by using a pre-trained ResNet-18 architecture and custom data augmentation techniques.Key Features:Transfer Learning: Fine-tuned ResNet-18 model for rapid convergence.Robust Augmentation: Includes RandomResizedCrop and RandomHorizontalFlip to improve model generalization.Validation Pipeline: Automated data splitting into training and validation sets.Visualization: Built-in scripts to visualize batch predictions and sample data.
## 🛠️ Installation & SetupClone the repo:
Bashgit clone https://github.com/your-username/dish-classifier.git
cd dish-classifier
Install dependencies:Bashpip install torch torchvision numpy matplotlib tqdm pandas
To run this project on your local machine, you will need Python 3 and the PyTorch ecosystem.

1. Prerequisites
First, ensure you have the necessary libraries installed. Open your terminal or command prompt and run:

Bash

pip install torch torchvision numpy matplotlib tqdm pandas
2. Dataset Preparation
The project is based on the Kaggle Platesv2 Dataset.

Download the dataset and extract the plates.zip file.

The notebook expects a specific folder structure. You should have a plates/ folder containing train/ and test/ subdirectories.

Inside train/, there should be two folders: cleaned/ and dirty/.

3. Execution Steps
Since the project is provided as a Jupyter Notebook (.ipynb), follow these steps:

Launch the Notebook:

Bash

jupyter notebook my-python-s-pytorch.ipynb
Setup Data Paths: In the first few cells, update the data path variables if your dataset is not in the default directory:

Python

# Example of updating path
data = './plates/' 
Run Data Organization: Execute the "Photo modification" section. This script will automatically split your training images into train and val (validation) sets to monitor model accuracy during training.

Train the Model:

Run the cells under "Building Torch's model". The script uses a pre-trained ResNet-18 model.

It applies data augmentations like RandomResizedCrop and RandomHorizontalFlip to help the model learn better.

View Results: The final cells will generate a submission.csv containing the model's predictions on the test set.

## 🛠️ Quick Tips
GPU Acceleration: If you have an NVIDIA GPU, the code is set to use cuda for significantly faster training.

Batch Size: The default batch size is set to 8. If you have a high-end GPU, you can increase this to speed up training.

Input Size: Images are resized to 224x224 for training and 244x244 for validation.

Dataset: Place the plates.zip file in the project directory or update the path in the notebook.
## 📊 Dataset StructureThe dataset is dynamically organized into the following structure during execution:Plaintextplates/
├── train/
│   ├── cleaned/
│   └── dirty/
├── val/
│   ├── cleaned/
│   └── dirty/
## 🧠 Model ArchitectureWe use a ResNet-18 backbone pre-trained on ImageNet.
The final fully connected layer is modified to output 2 classes (Clean/Dirty).Pythonmodel = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
## 📈 Training ConfigurationParameterValueBatch Size8Input Resolution244x244OptimizerAdam (default)SchedulerStepLR
## 🖼️ Sample VisualizationsThe model uses a normalization mean of [0.485, 0.456, 0.406] and standard deviation of [0.229, 0.224, 0.225] to align with ImageNet standards.
## 📜 LicenseDistributed under the MIT License. See LICENSE for more information.Tips for your GitHub:Add a Header Image: If you have a screenshot of your model results or a collage of clean/dirty dishes, add it at the very top.Kaggle Notebook Link: Since your code refers to /kaggle/input/, it's a great idea to add a link to your original Kaggle notebook in the "Overview" section.Requirements File: Create a requirements.txt file containing the libraries you imported (torch, torchvision, matplotlib, etc.).
