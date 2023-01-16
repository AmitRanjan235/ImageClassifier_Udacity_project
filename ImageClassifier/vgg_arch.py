from torchvision import models
from torch import nn
import torch




def build_network(arch, hidden_dim, output_dim, drop_prob):
    """
    arch: The architecture of the network. The options are 'vgg16' or 'vgg19'. If any other value is passed, 
    the function will use 'vgg16' by default.
    hidden_dim: The number of neurons in the hidden layer of the network.
    output_dim: The number of output categories.
    drop_prob: The dropout probability used during training.
    
    """
    if arch == 'vgg16':
        print("Using pretrained vgg16")
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        print("Using pretrained vgg19")
        model = models.vgg19(pretrained=True)
        
    else:
        print(f"this model architecture {arch} is not a valid model. Using vgg16 by default.")
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, output_dim),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

def get_optimizer(model, lr):
   return torch.optim.Adam(model.classifier.parameters(), lr)

def loss_function():
    return nn.NLLLoss()

def save_model(model, save_dir, arch, epochs, lr, hidden_units):
    """
    model: The trained model.
    save_dir: The directory where the model will be saved.
    arch: The architecture of the model. It can be 'vgg16' or 'vgg19'.
    epochs: The number of training epochs.
    lr: The learning rate used during training.
    hidden_units: The number of neurons in the hidden layer of the model.
    """
    if save_dir is '':
        save_path = f'./checkpoint-{arch}.pth'
    else:
        save_path = save_dir + f'/checkpoint-{arch}.pth'
    model.cpu()

    checkpoint = {
        'arch': arch,
        'hidden_dim': hidden_units,
        'epochs': epochs,
        'lr': lr,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_path)


def load_model(checkpoint_path):
    """
    checkpoint_path: The path to the checkpoint file containing the trained model's state dictionary and other    
    metadata such as the architecture and class to index mapping.
    
    """
    trained_model = torch.load(checkpoint_path)
       
    model = build_network(arch=trained_model['arch'], hidden_dim=trained_model['hidden_dim'],
    output_dim=102, drop_prob=0)  
    model.class_to_idx = trained_model['class_to_idx']
    model.load_state_dict(trained_model['state_dict'])
    print(f"vgg model loaded with arch {trained_model['arch']}")
    return model
