import torch
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
from daisee_model import DAiSEEModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, choices=['mse', 'l1', 'smoothl1'], required=True)

    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--affection', type=str,
                        choices=["Engagement", "Boredom", "Confusion", "Frustration"], required=True)
    parser.add_argument('--labels_dir', type=str, default='.')
    parser.add_argument('--subsample_rate', type=int, default=10)
    parser.add_argument('--out', action='store_true')
    parser.add_argument('--it', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)

    return parser.parse_args()


def test_model(model, criterion):
    since = time.time()
    dataset_sizes = {"Test": 1784}
    acc = 0.0
    device = torch.device('cuda')

    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    for sample_number, (inputs, labels) in enumerate(dataloaders['Test']):

        for k, v in inputs[0].items():
            v[0] = v[0].to(device).detach()
            v[1] = v[1].to(device).detach()
            v[2] = v[2].to(device).detach()

        labels = labels.to(device).detach()
        for k, v in inputs[1].items():
            v["bbox_img"] = v["bbox_img"].to(device).detach()
            v["egaze"] = v["egaze"].to(device).detach()
            v["lmarks"] = v["lmarks"].to(device).detach()

        with torch.set_grad_enabled(False):

            outputs = model(inputs)
            print(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        running_corrects += torch.sum(preds.data == torch.max(labels, 1)[1])

        print(sample_number, "running_loss", running_loss)

    final_loss = running_loss / dataset_sizes['Test']
    final_acc = running_corrects.double() / dataset_sizes['Test']
    print('Test loss: {:.4f} Acc {:.4f}'.format(final_loss, final_acc))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Accuracy: {:6f}'.format(final_acc))


if __name__ == '__main__':

    args = parse_args()

    dataloaders = {'Test': data.DataLoader(
        DAiSEEDataset(args.dataset_dir,
                      args.affection,
                      "Test",
                      args.subsample_rate
                      ),
        batch_size=1,
        shuffle=False,
        num_workers=32
    )}

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    log_dir = "it" + args.it
    if not osp.exists(log_dir):
        os.mkdir(log_dir)

    if args.out:
        sys.stdout = open(osp.join(log_dir, args.affection + ".txt"), mode='w+')

    losses = {
        "mse": nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smoothl1': nn.SmoothL1Loss()
    }

    model = DAiSEEModel(args.subsample_rate)
    assert osp.exists(args.weights)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    test_model(model, losses[args.loss])
