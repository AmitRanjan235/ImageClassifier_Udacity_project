from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loaders(data_dir="./flowers"):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms())
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms())
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms())
    class_to_idx = train_data.class_to_idx

    train_load = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_load = DataLoader(valid_data, batch_size=32, shuffle=True)
    test_load = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_load, valid_load, test_load, class_to_idx


def train_transforms():
    return transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    
def valid_transforms():
    return transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])    
    

def test_transforms():
    return transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
