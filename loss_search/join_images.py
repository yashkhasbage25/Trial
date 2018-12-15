from PIL import Image
import os.path as osp
import numpy as np
import sys

python3 join_images.py img1


optims = ["adam", "adadelta", "rmsprop", "sgd"]
losses = ["l1", "smoothl1", "rmse", "mse"]

img_dir = sys.argv[1]
if not osp.exists(img_dir):
    raise Exception("img dir not found")


def load_image(path):
    # with open(path, 'rb') as f:
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


img = np.vstack(np.hstack(
    (np.asarray(load_image(osp.join("img", optim+"_"+loss+".png"))) for loss in losses)) for optim in optims)
img = Image.fromarray(img)
img.save('optim_vs_loss.png')
