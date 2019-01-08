import os
import sys
import torch
import time
import numpy as np
import cv2
import torch.nn as nn
import torch.utils.data as data
import os.path as osp
from PIL import Image

import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import copy
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import *
from opts import parse_opts
from model import generate_model
from mean import get_mean
#from classify import classify_video
#from torchviz import make_dot
import pdb
# 50 : 40 + 10
# # TODO: change opts: add affection change default for mode
# and convert video data to frames. add dataset path in opts for this
# save weights and optimizer
# proper hyperparameters
# CUDA_VISIBLE_DEVICES=1 python3 train_model.py --loss mse --epochs 10 --optim adagrad --affection Boredom --it 104

class MyModel(nn.Module):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        self.resnet = generate_model(opt)
        self.resnet_data = torch.load(opt.model)
        self.resnet_state_dict = torch.load('resnet-34-kinetics.pth')
        self.resnet.load_state_dict(self.resnet_data['state_dict'])
        self.resnet.train()
        self.lin = nn.Linear(512, 4)
        self.smax = nn.Softmax(dim=1)
        
        device = torch.device('cuda')
        self.h0 = torch.randn(1, 1, 4).to(device)
        self.c0 = torch.randn(1, 1, 4).to(device)
        self.lstm = nn.LSTM(4, 4).to(device)
        

    def forward(self, x):
        self.h0.detach_()
        self.c0.detach_()
        for i in range(19):
            y = self.resnet(x[0][i])
            y = self.lin(y)
            y = self.smax(y).view(1, 1, 4)      
            lstm_outputs, (self.h0, self.c0) = self.lstm(y, (self.h0, self.c0))
            
        lstm_outputs = self.smax(lstm_outputs.view(1,4))
        return lstm_outputs
        

#def train_model(model, criterion, optimizer, scheduler, optim_name, loss_name, num_epochs=25):
def train_model(model, criterion, optimizer, scheduler, optim_name, loss_name, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_params = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    print("optim_name:", optim_name)
    print("loss_name:", loss_name)
    loss_list = {"Train": [], "Validation": []}
    device = torch.device('cuda')
    


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'epochs')
        ax.set_ylabel(r'loss')
        ax.set_title(optim_name+" + "+loss_name)

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                scheduler.step()
                #model.train()
                model.module.train()  # Set model to training mode
            else:
                #model.eval()
                model.module.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            resnet_outputs_list = []
            # Iterate over data.
            for sample_number, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):

                    outputs = model(inputs)

                    loss = criterion(outputs.view(1, 4), labels.to(device))
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                            
                        loss.backward()
                        optimizer.step()
                        #make_dot(loss, model.named_parameters())

                running_loss += loss.item()
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
                if sample_number%100 == 0:
                    print("sample", sample_number, "running_loss", running_loss)

                #del inputs, labels, loss

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            loss_list[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_opt_params = copy.deepcopy(optimizer.state_dict())

        plt.plot([i for i in range(len(loss_list["Train"]))], loss_list["Train"], label='train')
        plt.plot([i for i in range(len(loss_list["Validation"]))], loss_list["Validation"], label='val')            
        ax.set_ylim(bottom=0, top=1.2*max(max(loss_list["Train"]), max(loss_list["Validation"])))
        ax.legend(loc='best',fancybox=True,shadow=True)
        ax.grid()
        plt.tight_layout()
        plt.savefig(osp.join(imgs_dir, "{:}_{:}_{:}.png".format(opt.affection, optim_name, loss_name)))
        plt.clf()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    

    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_opt_params)
    torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "{:}_{:}_{:}_{:}.pth".format(opt.it, opt.affection, optimizer_name, loss_name))
    print('-'*10)


def one_hot(x):
    if x == 0:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif x == 1:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    elif x == 2:
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif x == 3:
        return np.array([0, 0, 0, 1], dtype=np.float32)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def pil_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def video_loader(video_path, spatial_transform):
    video = []
    
    assert osp.exists(video_path), print(video_path, "not found.")
    cap = cv2.VideoCapture(video_path)
    for i in range(300):
        ret, frame = cap.read()
        assert ret, "could not retrive frame"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        video.append(frame)
        if i == 299:
            video.append(frame.copy())
            video.append(frame.copy())
            video.append(frame.copy())
            video.append(frame.copy())
    cap.release()

    clip = np.array([spatial_transform(img).detach().numpy() for img in video])

    video = np.array([[clip[0:16]]])

    for i in range(1, 19):
        video = np.append(video, np.array([[clip[16*i:16*(i+1)]]]), axis=0)

    video = torch.FloatTensor(video)
    video = video.permute(0, 1, 3, 2, 4, 5)
    #clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    return video


class VideoLoader(data.Dataset):
    
    def __init__(self, root_dir, labels_csv, split_group, spatial_transforms):
        self.root_dir = root_dir
        self.labels_csv = labels_csv
        self.split_group = split_group
        self.spatial_transforms = spatial_transforms
        
        self.df = pd.read_csv(self.labels_csv)
        
    def __len__(self):

        return len(self.df)
        
    def __getitem__(self, i):
        
        row = self.df.iloc[i]
        video_path = row["Path"]
        label = row[opt.affection]
        clip = video_loader(video_path, self.spatial_transforms)
        return clip, one_hot(label)


opt = parse_opts()
opt.mean = get_mean()
opt.sample_size = 112  # transform images to this size
#opt.n_classes = 400
opt.sample_duration = 16
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCK'] = '1'
if not opt.no_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

split_groups = ["Train", "Validation"]
dataset_dir = opt.root_dir
labels_dir = opt.labels_dir

labels_csv = {}
for group in split_groups:
    labels_csv[group] = osp.join(labels_dir, group+"Labels.csv")

imgs_dir = "img"+str(opt.it)
if not osp.exists(imgs_dir):
    os.mkdir(imgs_dir)
sys.stdout = open(osp.join(imgs_dir, opt.affection+"_"+opt.optim+"_"+opt.loss+".txt"), mode='w+')

spatial_transform = Compose([Resize(opt.sample_size),
                             CenterCrop(opt.sample_size),
                             ToTensor(),
                             Normalize(opt.mean, [1, 1, 1])])

dataloaders = {}  # FIXME:
dataloaders["Train"] = data.DataLoader(VideoLoader(
    opt.root_dir, labels_csv["Train"], "Train", spatial_transform), batch_size=1, shuffle=True, num_workers=32)
dataloaders["Validation"] = data.DataLoader(VideoLoader(
    opt.root_dir, labels_csv["Validation"], "Validation", spatial_transform), batch_size=1, shuffle=True, num_workers=8)

opt.mean = get_mean()
dataset_sizes = {}
dataset_sizes["Train"] = len(dataloaders["Train"])
dataset_sizes["Validation"] = len(dataloaders["Validation"])
device = torch.device('cuda')

# losses = ["mse", "l1", "l1smooth", "rmse"]
losses = {"mse": nn.MSELoss(), "l1": nn.L1Loss(), "smoothl1": nn.SmoothL1Loss(), "rmse": RMSELoss()}
losses = {opt.loss: losses[opt.loss]}

#optimizer_names = ["adam", "sgd", "adadelta", "rmsprop"]
optimizer_names = [opt.optim]

for loss_name, loss_func in losses.items():
    for optimizer_name in optimizer_names:
        my_model = MyModel(opt)
        #my_model.cuda()
        
        #
        my_model.to(device)
        my_model = nn.DataParallel(my_model)
        #pdb.set_trace()
        if optimizer_name == "adam":
            optimizer = optim.Adam(my_model.parameters(), lr=0.000005, weight_decay=0.1)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(my_model.parameters(), lr=0.5, momentum=0.98, weight_decay=0.1)
        elif optimizer_name == 'adadelta':
            optimizer = optim.Adadelta(my_model.parameters(), lr=0.000005, weight_decay=0.1)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(my_model.parameters(), lr=0.000005, momentum=0.98, weight_decay=0.1)
        elif optimizer_name == 'adagrad':
            optimizer = optim.Adagrad(my_model.parameters(), lr=0.00005, weight_decay=0.1)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        optimizer.zero_grad()
        train_model(
            my_model, loss_func, optimizer, scheduler, optimizer_name, loss_name, num_epochs=opt.epochs)
        # torch.save(best_model.state_dict({'state_dict': best_model.state_dict(),
        #                                  'optimizer': optimizer_dict}
        #                                 ), "{}_{}.pth".format(optimizer_name, loss_name))
