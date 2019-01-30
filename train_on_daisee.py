import torch
import copy
import os
import time
import argparse
import sys
import pdb

import os.path as osp
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from load_dataset import DAiSEEDataset
from daisee_model import DAiSEEMicroExpressionModel

# CUDA_VISIBLE_DEVICES=2,3 python3 train_on_daisee.py --loss smoothl1 --optim adam --it 201 --epochs 1000000 --subsample_rate 10 --affection Confusion
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, choices=['mse', 'l1', 'smoothl1', "ce"], required=True)
    parser.add_argument('--optim', type=str,
                choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rmsprop'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--affection', type=str,
                choices=["Engagement", "Boredom", "Confusion", "Frustration"], required=True)
    parser.add_argument('--labels_dir', type=str, default='.')
    parser.add_argument('--subsample_rate', type=int, default=10)
    parser.add_argument('--it', type=str, required=True)
    parser.add_argument('--out', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--cmnt', type=str, default=None)

    return parser.parse_args()

class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        assert yhat.shape == (1, 4)
        assert y.shape == (1, 4)
        return torch.mean(torch.sum((yhat-y)**2))

class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        assert yhat.shape == (1, 4)
        assert y.shape == (1, 4)
        return torch.mean(torch.abs(yhat - y))

def train_model(model, criterion, optimizer, scheduler, optim_name, loss_name, num_epochs=25, weights=None):
#def train_model(model, criterion, optimizer, optim_name, loss_name, num_epochs=25, weights=None):
    since = time.time()
    if args.cmnt:
    	print(args.cmnt)
    dataset_sizes = {"Train":5358, "Validation": 1429}
    if weights:
        assert osp.exists(weights)
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_params = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    print("optim_name:", optim_name)
    print("loss_name:", loss_name)
    loss_list = {"Train": [], "Validation": []}
    device = torch.device('cuda') 

    for epoch in range(num_epochs):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlabel(r'epochs')
        ax.set_ylabel(r'loss')
        ax.set_title(optim_name+" + "+loss_name)
        ax.set_ylim(bottom=0)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss=0.0
            running_corrects=0

            # Iterate over data.
            for sample_number, (inputs, labels) in enumerate(dataloaders[phase]):
                # inputs[0]=inputs[0].to(device)
                for k, v in inputs[0].items():
                    v[0] = v[0].to(device).detach()
                    v[1] = v[1].to(device).detach()
                    v[2] = v[2].to(device).detach()
                labels=labels.to(device).detach()
                for k, v in inputs[1].items():
                    v["bbox_img"] = v["bbox_img"].to(device).detach()
                    v["egaze"] = v["egaze"].to(device).detach()
                    v["lmarks"] = v["lmarks"].to(device).detach()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):

                    outputs=model(inputs)
                    
                    _, preds=torch.max(outputs, 1)
                    #print(labels, torch.max(labels,1)[1])
                    loss=criterion(outputs, torch.max(labels,1)[1])
                    #loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                #if phase == 'Train':
                #    running_loss += loss.item()
                #else:
                #    running_loss += loss.item()
                running_loss += loss.item()
                running_corrects += torch.sum(preds.data == torch.max(labels, 1)[1])
                #if sample_number % 200 == 0:
                print(sample_number, "running_loss", running_loss)
                #print(sample_number)
                #print(sample_number, "running_loss", running_loss)
            epoch_loss=running_loss / dataset_sizes[phase]
            epoch_acc=running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            loss_list[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
                best_opt_params = copy.deepcopy(optimizer.state_dict())
            plt.plot([i for i in range(len(loss_list["Train"]))], loss_list["Train"], label = 'Train')
            plt.plot([i for i in range(len(loss_list["Validation"]))], loss_list["Validation"], label = 'Validation')
            time_elapsed=time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights

            plt.legend(loc = 'best', fancybox = True, shadow = True)
            ax.grid()
            plt.tight_layout()
            plt.savefig(osp.join(log_dir, "{:}_{:}_{:}.png".format(
                args.affection, optim_name, loss_name)))
            plt.clf()
            print('-'*10)
            
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_opt_params)
        torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "{:}_{:}_{:}_{:}.pth".format(args.it, args.affection, optimizer_name, loss_name))
        print('-'*10)
    # print(loss_list["train"])
    # print(loss_list["val"])
    # plt.plot([i for i in range(len(loss_list["train"]))], loss_list["train"], 'bo-', label='train')
    # plt.plot([i for i in range(len(loss_list["val"]))], loss_list["val"], 'ro-', label='val')
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    #
    # # load best model weights
    # plt.savefig(osp.join(imgs_dir, "{:}_{:}_{:}.png".format(opt.affection, optim_name, loss_name)))
    # plt.legend(loc='best')
    # ax.grid()
    # plt.tight_layout()
    # plt.clf()

    #
    # model.load_state_dict(best_model_wts)


if __name__ == '__main__':
    pdb.set_trace()
    args=parse_args()
    dataloaders={
            "Train": data.DataLoader(
                                        DAiSEEDataset(args.dataset_dir,
                                        args.affection, "Train", args.subsample_rate),
                        batch_size=1, shuffle=False,
                        num_workers=1),
            "Validation": data.DataLoader(
                                        DAiSEEDataset(args.dataset_dir,
                                        args.affection, "Validation", args.subsample_rate),
                        batch_size=1, shuffle=False,
                        num_workers=1)
        }


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    log_dir = "it" + args.it
    if not osp.exists(log_dir):
        os.mkdir(log_dir)
    if args.out:
        sys.stdout = open(osp.join(log_dir, "%s_%s_%s.txt"%(args.affection, args.loss, args.optim)), mode='w+')
    # dataset_dir = args.dataset_dir
    losses = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(), 
        "smoothl1": nn.SmoothL1Loss(),
        "ce": nn.CrossEntropyLoss(weight=torch.FloatTensor([1,3,5,50]).cuda())
    }
    optimizer_name = args.optim
    loss_name = args.loss
    loss_func = losses[args.loss]

    model = DAiSEEMicroExpressionModel(10, 300)
    model = model.cuda()
    num_epochs = args.epochs

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.001)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.00000025, weight_decay=0.1)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=0.000025, weight_decay=0.1)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.00000025, momentum=0.98, weight_decay=0.1)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0, weight_decay=0.1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    optimizer.zero_grad()
    #train_model(model, loss_func, optimizer, optimizer_name, loss_name, num_epochs, weights=args.weights)
    train_model(model, loss_func, optimizer, scheduler, optimizer_name, loss_name, num_epochs, weights=args.weights)
