import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    # opt.n_classes = 400

    if not opt.no_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model = generate_model(opt)

    print('loading model {}'.format(opt.model))
    if opt.my_pc:
        model_data = torch.load(opt.model, map_location='cpu')
    else:
        model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    # state_dict = torch.load('resnet-34-kinetics.pth')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_data['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        # print(k)
    # load params
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
            subprocess.call('mkdir tmp', shell=True)
            subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
                            shell=True)

            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)

            subprocess.call('rm -rf tmp', shell=True)
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
