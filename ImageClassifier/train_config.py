import argparse
import utils
import vgg_arch


def training_config(args):
    print("Training Configuration of the vgg model:")
    print(f"Architecture of model: {args.arch}")
    print(f"Initial learning rate for model: {args.learning_rate}")
    print(f"Number of neurons in hidden layers of vgg architecture: {args.hidden_units}")
    print(f"Number of output classes: {args.output_units}")
    print(f"Dropout probability: {args.drop_prob}")
    print(f"Number of training Epochs for the model: {args.epochs}")
    print(f"Using GPU for inference: {args.gpu}")