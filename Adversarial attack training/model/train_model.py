import torch
import torch.nn as nn
import torch.optim as optim
import os
from data.dataloader import dataloader
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F


NUM_EPOCHS = 50
epsilon = 0.005
k = 3
alpha = 0.005
lr = 0.001
csv_file = 'D:/PyTorch/Adversarial Attack Training/csv/'


class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.rand_like(x) * 2 * epsilon - epsilon
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

class FGMAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x.requires_grad_()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x + epsilon * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

class FSGMAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x.requires_grad_()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]

        x_adv = x + epsilon * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

class FreeLBAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.rand_like(x) * 2 * epsilon - epsilon
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x


def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

def prepare_trained_model(model):

    if os.path.isfile("./traned_model"):
        print("no need to train.")
        model.load_state_dict(torch.load("./trained_model"))
        return model
    else:
        train(model)
        return model

def train(model):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    df.to_csv(csv_file + 'PGD-{}-{}-{}-{}.csv'.format(NUM_EPOCHS,epsilon,k,alpha))

    train_loader, val_loader, test_loader = dataloader()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    #pgd_attack = LinfPGDAttack(model)
    #adv_pgd = attack(x, y, model, pgd_attack)

    #fgm_attack = FGMAttack(model)
    #adv_fgm = attack(x, y, model, fgm_attack)

    #fsgm_attack = FSGMAttack(model)
    #adv_fsgm = attack(x, y, model, fsgm_attack)

    #freelb_attack = FreeLBAttack(model)
    #adv_freelb = attack(x, y, model, freelb_attack)

    adversary = LinfPGDAttack(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_total = 0
        train_loss = 0
        train_accuracy = 0.0
        train_adv_loss = 0.0
        train_clean_loss = 0.0



        print(f'| Epoch: {epoch + 1}')

        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            train_clean_optputs = model(imgs)
            train_clean_loss = criterion(train_clean_optputs, labels)
            train_clean_loss.backward()

            train_adv = adversary.perturb(imgs, labels)
            train_adv_outputs = model(train_adv)
            train_adv_loss = criterion(train_adv_outputs, labels)
            train_adv_loss.backward()

            optimizer.step()

            train_clean_predicted = torch.argmax(train_clean_optputs, 1)
            train_clean_accuracy = torch.sum(train_clean_predicted == labels.to(device))
            train_adv_predicted = torch.argmax(train_adv_outputs, 1)
            train_adv_accuracy = torch.sum(train_adv_predicted == labels.to(device))

            train_adv_loss += abs(train_adv_loss.item()) * imgs.size(0)
            train_clean_loss += train_clean_loss.item() * imgs.size(0)
            train_loss += abs(train_adv_loss.item()) * imgs.size(0) + train_clean_loss.item() * imgs.size(0)
            train_accuracy += (train_clean_accuracy + train_adv_accuracy)

            train_total += labels.size(0)

        torch.cuda.empty_cache()
        print(train_total, len(train_loader))
        train_total_loss = train_loss / (train_total*2)
        train_total_accuracy = train_accuracy / (len(train_loader)*2)

        print(f'| Train Clean Loss: {train_clean_loss:.4f} | Train Adversarial Loss: {train_adv_loss:.4f}')
        print(f'| Train Loss: {train_total_loss:.4f} | Train Acc: {train_total_accuracy * 100:.3f}%')

        model.eval()
        val_correct, val_total, val_loss, valval_total_loss, val_total_accuracy = 0, 0, 0, 0, 0

        with torch.no_grad():
            val_accuracy_clean = 0.0
            val_accuracy_adv = 0.0
            val_adv_loss = 0.0
            val_clean_loss = 0.0
            for batch in tqdm(val_loader):
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

                val_clean_optputs = model(imgs)
                val_clean_loss = criterion(val_clean_optputs, labels)

                val_adv = adversary.perturb(imgs, labels)
                val_adv_outputs = model(val_adv)
                val_adv_loss = criterion(val_adv_outputs, labels)

                val_clean_predicted = torch.argmax(val_clean_optputs, 1)
                val_clean_accuracy = torch.sum(val_clean_predicted == labels.to(device)).item() / labels.size(0)
                val_adv_predicted = torch.argmax(val_adv_outputs, 1)
                val_adv_accuracy = torch.sum(val_adv_predicted == labels.to(device)).item() / labels.size(0)

                val_adv_loss += abs(val_adv_loss.item()) * imgs.size(0)
                val_clean_loss += val_clean_loss.item() * imgs.size(0)
                val_loss += abs(val_adv_loss.item()) * imgs.size(0) + val_clean_loss.item() * imgs.size(0)
                val_accuracy_clean += val_clean_accuracy
                val_accuracy_adv += val_adv_accuracy

                val_total += labels.size(0)

            torch.cuda.empty_cache()
            print(val_total, len(val_loader))
            val_total_loss = val_loss / (val_total*2)
            val_total_accuracy = (val_accuracy_adv + val_accuracy_clean) / (len(val_loader)*2)

            print(f'| Val Clean Loss: {val_clean_loss:.4f} | Val Adversarial Loss: {val_adv_loss:.4f}')
            print(f'| Val Loss: {val_total_loss:.4f} | Val Acc: {val_total_accuracy * 100:.4f}%')
            #print(f"Last, Best Acc. [{best_acc:.5f}]")
            #print('=' * 100)

        model.eval()
        test_correct, test_total, test_loss, test_total_loss, test_total_accuracy = 0, 0, 0, 0, 0

        with torch.no_grad():
            test_accuracy_clean = 0.0
            test_clean_loss = 0.0
            for batch in tqdm(test_loader):
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

                test_clean_optputs = model(imgs)
                test_clean_loss = criterion(test_clean_optputs, labels)

                test_clean_predicted = torch.argmax(test_clean_optputs, 1)
                test_clean_accuracy = torch.sum(test_clean_predicted == labels.to(device)).item() / labels.size(0)

                #test_clean_loss += test_clean_loss.item() * imgs.size(0)
                test_loss +=  test_clean_loss.item() * imgs.size(0)
                test_accuracy_clean += test_clean_accuracy
                test_total += labels.size(0)

            torch.cuda.empty_cache()
            test_total_loss = test_loss / test_total
            print(test_total,len(test_loader))
            test_total_accuracy =test_accuracy_clean / len(test_loader)

        if test_total_accuracy > best_acc:
            best_acc = test_total_accuracy
            torch.save(model.state_dict(), './trained_model')

        #print(f'| Test Clean Loss: {test_clean_loss:.4f} | Test Adversarial Loss: {test_adv_loss:.4f}')
        print(f'| Test Loss: {test_total_loss:.4f} | Test Acc: {test_total_accuracy * 100:.4f}%')

        print('=' * 100)

        jc = "%d" % (epoch + 1)
        tr_adv_loss = '%.4f' % train_adv_loss
        tr_cle_loss = '%.4f' % train_clean_loss
        tr_total_loss = '%.4f' % train_total_loss
        tr_total_acc = '%.4f' % (train_total_accuracy * 100)

        v_adv_loss = '%.4f' % val_adv_loss
        v_cle_loss = '%.4f' % val_clean_loss
        v_total_loss = '%.4f' % val_total_loss
        v_total_acc = '%.4f' % (val_total_accuracy * 100)

        #te_adv_loss = '%.4f' % test_adv_loss
        te_cle_loss = '%.4f' % test_loss
        #te_total_loss = '%.4f' % test_total_loss
        te_total_acc = '%.4f' % (test_accuracy_clean * 100)

        list1 = [jc,tr_adv_loss,tr_cle_loss, tr_total_loss, tr_total_acc,
                 v_adv_loss,v_cle_loss, v_total_loss, v_total_acc,
                te_cle_loss,  te_total_acc]
        data = pd.DataFrame([list1])
        data.to_csv(csv_file + 'PGD-{}-{}-{}-{}.csv'.format(NUM_EPOCHS,epsilon,k,alpha), mode='a', header=False, index=False)
    print(f"Last, Best Acc. [{best_acc:.5f}]")

    return model

