#importing required libraries
import argparse
import utils
import vgg_arch
import torch
import torch.optim as optim
import time
from torchvision import models
from validation import validate
from train_config import training_config 


#defining a function to get arguments from command line
def get_args():
    """ 
    data_dir: The path to the directory containing the dataset for flower images.
    save_dir: The directory where the VGG model checkpoints will be saved.
    arch: The deep neural network architecture used for training. The options are 'vgg16' or 'vgg19'.
    learning_rate: The learning rate used for training.
    hidden_units: The number of neurons in the hidden layer of the VGG architecture.
    output_units: The number of output categories.     
    drop_prob: The dropout probability used during training
    epochs: The number of training epoch.
    gpu: A flag indicating whether or not to use a GPU for training.
        
    """
    parser = argparse.ArgumentParser(description="Make predictions using a pre-trained model")
    parser.add_argument('data_dir',default='/home/workspace/ImageClassifier/flowers', type=str, help="The path to the directory containing the dataset for flower images.")
    parser.add_argument('--save_dir', default='/home/workspace/saved_model', type=str, help="The directory where the VGG model checkpoints will be saved.")
    parser.add_argument('--arch', default='vgg16',
    help='Deep NN architecture, options: vgg16, vgg19')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='The learning rate used for trainingThe learning rate used for training')
    parser.add_argument('--hidden_units', default=1024, type=int, help='The number of neurons in the hidden layer of the VGG architecture.')
    parser.add_argument('--output_units', default=102, type=int, help='The number of output categories')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='he dropout probability used during training')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='A flag indicating whether or not to use a GPU for training')
    return parser.parse_args()





def train_vgg(model, train_data, valid_data, criterion, optimizer, epochs, log_interval,gpu):
    if gpu and torch.cuda.is_available():
        print("Using GPU")
        model.cuda()

    steps = 0
    for e in range(epochs):
        running_loss = 0
        print(f"Epoch {e + 1} ------------------------------------")
        for images, labels in train_data:
            steps += 1
            if gpu and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % log_interval == 0:
                model.eval()
                t_loss = running_loss / log_interval
                v_loss, v_acc = validate(model, valid_data, criterion, gpu)
                print(f"Training Loss: {t_loss:.4f} Validation Loss: {v_loss:.4f} Validation Acc: {v_acc:.4f}")
                running_loss = 0
                model.train()


                



def main():
    args = get_args()
    training_config(args)
    train_data, valid_data, test_load, class_to_idx =utils.data_loaders(args.data_dir)
    model = vgg_arch.build_network(args.arch, args.hidden_units, args.output_units, args.drop_prob)
    model.class_to_idx = class_to_idx
    criterion = vgg_arch.loss_function()
    optimizer = vgg_arch.get_optimizer(model, args.learning_rate)
    train_vgg(model, train_data, valid_data, criterion, optimizer, args.epochs, 10, args.gpu)
    vgg_arch.save_model(model, args.save_dir, args.arch, args.epochs, args.learning_rate, args.hidden_units)

    
if __name__ == '__main__':
    main()
    
