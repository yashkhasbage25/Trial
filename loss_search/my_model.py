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

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import copy
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import *
from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

# 50 : 40 + 10


class MyModel(nn.Module):

    def __init__(self, opt):
        super(MyModel, self).__init__()
        self.resnet = generate_model(opt)
        self.resnet_data = torch.load(opt.model)
        self.resnet_state_dict = torch.load('resnet-34-kinetics.pth')
        self.resnet.load_state_dict(self.resnet_data['state_dict'])
        self.resnet.train()
        self.lin = nn.Linear(400, 4)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.lin(x)
        x = self.smax(x)
        return x


def train_model(model, criterion, optimizer, scheduler, optim_name, loss_name, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print("optim_name:", optim_name)
    print("loss_name:", loss_name)
    loss_list = {"train": [], "val": []}
    device = torch.device('cuda')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'epochs')
    ax.set_ylabel(r'loss')
    ax.set_title(optim_name+" + "+loss_name)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase == 'train':
                    running_loss += loss.item() * 40
                else:
                    running_loss += loss.item() * 10
                running_corrects += torch.sum(preds.data == torch.max(labels, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            loss_list[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    #print(loss_list["train"])
    #print(loss_list["val"])
    plt.plot([i for i in range(len(loss_list["train"]))], loss_list["train"], 'bo-', label='train')
    plt.plot([i for i in range(len(loss_list["val"]))], loss_list["val"], 'ro-', label='val')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    plt.savefig(osp.join(imgs_dir, "{:}_{:}.png".format(optim_name, loss_name)))
    plt.legend(loc='best')
    ax.grid()
    plt.tight_layout()
    plt.clf()
    #model.load_state_dict(best_model_wts)
    print('-'*10)
    #return model, optimizer.state_dict()
    #return None, None

def one_hot(x):
    if x == "0":
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif x == "1":
        return np.array([0, 1, 0, 0], dtype=np.float32)
    elif x == "2":
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif x == "3":
        return np.array([0, 0, 0, 1], dtype=np.float32)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


opt = parse_opts()
opt.mean = get_mean()
opt.sample_size = 112 # transform images to this size
opt.n_classes = 400
opt.sample_duration = 16

if not opt.no_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
dataset_train = []
dataset_test = []
dataset_dict = {"Train": {}, "Test": {}}

for group in ["Train", "Test"]:
    with open("My%s.txt" % group) as f:
        for line in f:
            (key, val) = line.split()
            dataset_dict[group][int(key)] = one_hot(val)
            if group == "Train":
                dataset_train.append(int(key))
            else:
                dataset_test.append(int(key))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'r') as f:
    #    with Image.open(f) as img:
    #        return img.convert('RGB')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def video_loader(video_dir_path, spatial_transform):
    video = []
    frame_indices = [i for i in range(1, 300) if i % 18 == 0]
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, "%d.jpg" % i)
        video.append(pil_loader(image_path))
    clip = video
    clip = [spatial_transform(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    assert clip.shape == (3, 16, 112, 112)
    return clip


class VideoLoader(data.Dataset):

    def __init__(self, root_dir, phase, spatial_transform):
        self.root_dir = root_dir
        self.phase = phase
        self.spatial_transform = spatial_transform

    def __len__(self):
        if self.phase == 'train':
            return 40
        else:
            return 10

    def __getitem__(self, index):
        if self.phase == 'train':
            video_dir_path = dataset_train[index]
            video_dir_path = osp.join(self.root_dir, str(video_dir_path))
            return video_loader(video_dir_path, self.spatial_transform), dataset_dict["Train"][dataset_train[index]]
        else:
            video_dir_path = dataset_test[index]
            video_dir_path = osp.join(self.root_dir, str(video_dir_path))
            return video_loader(video_dir_path, self.spatial_transform), dataset_dict["Test"][dataset_test[index]]

imgs_dir = "img2"
if not osp.exists(imgs_dir):
    os.mkdir(imgs_dir)

if not opt.no_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
spatial_transform = Compose([Resize(opt.sample_size),
                             CenterCrop(opt.sample_size),
                             ToTensor(),
                             Normalize(opt.mean, [1, 1, 1])])
dataloaders = {}
dataloaders["train"] = data.DataLoader(VideoLoader("dummy", "train", spatial_transform))
dataloaders["val"] = data.DataLoader(VideoLoader("dummy", "val", spatial_transform))

opt.mean = get_mean()
dataset_sizes = {}
dataset_sizes["train"] = len(dataloaders["train"])
dataset_sizes["val"] = len(dataloaders["val"])


# losses = ["mse", "l1", "l1smooth", "rmse"]
losses = {"mse": nn.MSELoss(), "l1": nn.L1Loss(), "smoothl1": nn.SmoothL1Loss(), "rmse": RMSELoss()}
losses = {opt.loss:losses[opt.loss]}
optimizer_names = ["adam", "sgd", "adadelta", "rmsprop"]


for loss_name, loss_func in losses.items():
    for optimizer_name in optimizer_names:
        my_model = MyModel(opt).cuda()
        if optimizer_name == "adam":
            optimizer = optim.Adam(my_model.parameters())
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(my_model.parameters(), lr=1e-3, momentum=0.98)
        elif optimizer_name == 'adadelta':
            optimizer = optim.Adadelta(my_model.parameters())
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(my_model.parameters(), lr=1e-3, momentum=0.98)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        optimizer.zero_grad()
        train_model(
            my_model, loss_func, optimizer, scheduler, optimizer_name, loss_name, num_epochs=opt.epochs)
        #torch.save(best_model.state_dict({'state_dict': best_model.state_dict(),
        #                                  'optimizer': optimizer_dict}
        #                                 ), "{}_{}.pth".format(optimizer_name, loss_name))
