# AI Programming with Python Project

Project Overview
This project is a part of the Udacity AI Programming with Python Nanodegree Program. In this project, we built an image classifier that can classify flower species among 102 different categories using a deep learning model. We used PyTorch to train an image classifier on a dataset of flower images.
Development Notebook
Package Imports
All the necessary packages and modules are imported in the first cell of the notebook.

Training Data Augmentation
torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping.

Data Normalization
The training, validation, and testing data is appropriately cropped and normalized.

Data Loading
The data for each set (train, validation, test) is loaded with torchvision's ImageFolder.

Data Batching
The data for each set is loaded with torchvision's DataLoader.

Pretrained Network
A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen.

Feedforward Classifier
A new feedforward network is defined for use as a classifier using the features as input.

Training the Network
The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static. During training, the validation loss and accuracy are displayed.

Testing Accuracy
The network's accuracy is measured on the test data.

Saving the Model
The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary.

Loading Checkpoints
There is a function that successfully loads a checkpoint and rebuilds the model.

Image Processing
The process_image function successfully converts a PIL image into an object that can be used as input to a trained model.

Class Prediction
The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image.

Sanity Checking with Matplotlib
A Matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names.

Command Line Application
Training a Network
train.py successfully trains a new network on a dataset of images. The training loss, validation loss, and validation accuracy are printed out as the network trains.

Model Architecture
The training script allows users to choose from at least two different architectures available from torchvision.models.

Model Hyperparameters
The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs.

Training with GPU
The training script allows users to choose training the model on a GPU.

Predicting Classes
The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and its associated probability.

Top K Classes
The predict.py script allows users to print out the top K classes along with associated probabilities.

Displaying Class Names
The predict.py script allows users to load a JSON file that maps the class values to other category names.

Predicting with GPU
The predict.py script allows users to use the GPU to calculate the predictions.

Installation
Clone the repository:
bash
Copy code
git clone https://github.com/<username>/<repository>.git
Install the required packages:
Copy code
pip install -r requirements.txt
Train the model:
css
Copy code
python train.py data_dir --arch vgg16 --learning_rate 0.001 --hidden_units 1024 --output_units 102 --drop_prob 0.2 --epochs 10 --gpu
Replace data_dir with the path to the directory containing the dataset for flower images.


