import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from data.dataloader import dataloader


def eval_loss(model):
    _, test_loader = dataloader()
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    print(device)

    correct = 0
    total_loss = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = inputs.size(0)
            print(batch_size)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

        print(total)

    return total_loss / total, 100. * correct / total
