<h1>Flower Species Image Classifier Project</h1>
<h2>Project Overview</h2>
<p>This project is a part of the Udacity AI Programming with Python Nanodegree Program. In this project, we built an image classifier that can classify flower species among 102 different categories using a deep learning model. We used PyTorch to train an image classifier on a dataset of flower images.</p>

<h2>Part 1 - Development Notebook</h2>
<p>The development notebook contains all the code needed to train the flower image classifier from scratch. It includes the following components:</p>
<ul>
    <li>Package Imports: All the necessary packages and modules are imported in the first cell of the notebook.</li>
    <li>Training Data Augmentation: torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping.</li>
    <li>Data Normalization: The training, validation, and testing data is appropriately cropped and normalized.</li>
    <li>Data Loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder.</li>
    <li>Data Batching: The data for each set is loaded with torchvision's DataLoader.</li>
    <li>Pretrained Network: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen.</li>
    <li>Feedforward Classifier: A new feedforward network is defined for use as a classifier using the features as input.</li>
    <li>Training the Network: The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static.</li>
    <li>Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed.</li>
    <li>Testing Accuracy: The network's accuracy is measured on the test data.</li>
    <li>Saving the Model: The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary.</li>
    <li>Loading Checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model.</li>
    <li>Image Processing: The process_image function successfully converts a PIL image into an object that can be used as input to a trained model.</li>
    <li>Class Prediction: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image.</li>
    <li>Sanity Checking with Matplotlib: A Matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names.</li>
</ul>
<h2>Part 2 - Command Line Application</h2>
<p>The command line application allows the user to train and use the flower image classifier from the command line. It includes the following components:</p>
<ul>
    <li>Training a Network: train.py successfully trains a new network on a dataset of images.</li>
    <li>Training Validation Log: The training loss, validation loss, and validation accuracy are printed out as a network trains.</li>
    <li>Model Architecture: The training script allows users to choose from at least two different architectures available from torchvision.models.</li>
    <li>Model Hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs.</li>
    <li>Training with GPU: The training script allows users to choose training the model on a GPU.</li>
    <li>Predicting Classes: predict.py successfully reads in an image and a checkpoint then prints the most likely image class and its associated probability.</li>
    <li>Top K Classes: The predict.py script allows users to print out the top K classes along with associated probabilities.</li>
    <li>Displaying Class Names: The predict.py script allows users to load a JSON file that maps the class values to other category names.</li>
    <li>Predicting with GPU: The predict.py script allows users to use the GPU to calculate the predictions.</li>
</ul>

<h2>Installation</h2> 
<div>
  <p>To get started with this project, follow these steps:</p>
  <ol>
    <li>Clone the repository:</li>
  </ol>
  <pre><code class="language-bash">git clone https://github.com/AmitRanjan235/ImageClassifier_Udacity_project.git</code></pre>
  <ol start="2">
    <li>Install the required packages:</li>
  </ol>
  <pre><code class="language-bash">pip install -r requirements.txt</code></pre>
  <ol start="3">
    <li>Train the model:</li>
  </ol>
  <p>Run the following command to train the model:</p>
  <pre><code class="language-bash">python train.py data_dir --arch vgg16 --learning_rate 0.001 --hidden_units 1024 --output_units 102 --drop_prob 0.2 --epochs 10 --gpu</code></pre>
  <p>Replace <code>data_dir</code> with the path to the directory containing the dataset for flower images.</p>
</div>
