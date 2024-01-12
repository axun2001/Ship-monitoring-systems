from model.alexnet import AlexNet
from model.resnet import *
from model.train_model import prepare_trained_model
from directions import create_random_directions
from calc_loss import calulate_loss_landscape


if __name__ == '__main__':
    #model = AlexNet()
    #model = ResNet18()
    model = ResNet34()

    rand_directions = create_random_directions(model)
    trained_model = prepare_trained_model(model)
    calulate_loss_landscape(trained_model, rand_directions)

