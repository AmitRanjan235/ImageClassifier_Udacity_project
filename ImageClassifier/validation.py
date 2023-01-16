import utils
import vgg_arch
import torch
from torchvision import models


def validate(model, data_loader, criterion, gpu):
    if gpu and torch.cuda.is_available():
        model.to('cuda')

    loss = 0
    acc = 0

    for images, labels in data_loader:
        if gpu and torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')

        with torch.no_grad():
            outputs = model.forward(images)
            loss += criterion(outputs, labels)
            preds = torch.exp(outputs).data

            equality = (labels.data == preds.max(1)[1])
            acc += equality.type_as(torch.FloatTensor()).mean()

    loss /= len(data_loader)
    acc /= len(data_loader)

    return loss, acc