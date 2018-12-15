import argparse
import pandas as pd
import os.path as osp
import shutil as sh
import os
import cv2


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--main', type=str, help='path to main dir')
    parser.add_argument('-d', '--dummy', type=str, help='path to dummy dataset dir')

    return parser.parse_args()


def create_frames(video_path, video_name):
    new_dir = osp.join(args.dummy, video_name)
    if not osp.exists(new_dir):
        os.mkdir(new_dir)
    
    cap = cv2.VideoCapture(video_path)
    for i in range(1, 301):
        ret, frame = cap.read()
        if i % 18 == 0:
            #print(video_path)
            cv2.imwrite(osp.join(new_dir, "%d.jpg" % i), frame)


args = build_parser()
all_df = pd.read_csv(osp.join(args.main, 'Features', 'AllLabels.csv'))

if not osp.exists(args.dummy):
    os.mkdir(args.dummy)
else:
    sh.rmtree(args.dummy)
    os.mkdir(args.dummy)

sampled_df = all_df.sample(n=50)
with open("MyTrain.txt", mode='w+') as f:
    for i in range(40):
        row = sampled_df.iloc[i]
        f.write("%s %d\n" % (osp.splitext(row["ClipID"])[0], row["Engagement"]))
        create_frames(row["Path"], osp.splitext(row["ClipID"])[0])

with open("MyTest.txt", mode='w+') as f:
    for i in range(40, 50):
        row = sampled_df.iloc[i]
        f.write("%s %d\n" % (osp.splitext(row["ClipID"])[0], row["Engagement"]))
        create_frames(row["Path"], osp.splitext(row["ClipID"])[0])
