import torch
import cv2
import copy

import numpy as np
import pandas as pd
import os.path as osp
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

# # TODO: detach????


def get_useful_lmark_points():
    useful_lmarks = []
    for i in range(7, 12):
        useful_lmarks.append("x%d" % i)
        useful_lmarks.append("y%d" % i)
    for i in range(18, 28):
        useful_lmarks.append("x%d" % i)
        useful_lmarks.append("y%d" % i)
    for i in range(37, 69):
        useful_lmarks.append("x%d" % i)
        useful_lmarks.append("y%d" % i)
    return useful_lmarks


class BasicDataset:

    def __init__(self, dataset_dir, split_group):

        self.base_dir = dataset_dir
        self.dataset_dir = osp.join(self.base_dir, "DataSet")
        self.features_dir = osp.join(dataset_dir, "Features")
        assert osp.exists(self.dataset_dir)
        assert osp.exists(self.features_dir)

        self.split_group = split_group

        self.useful_lmarks = get_useful_lmark_points()

        self.bbox_dir = osp.join(self.features_dir, "BBox")
        self.lmarks_dir = osp.join(self.features_dir, "LMarks")
        self.egaze_dir = osp.join(self.features_dir, "EGaze")
        assert osp.exists(self.bbox_dir)
        assert osp.exists(self.lmarks_dir)
        assert osp.exists(self.egaze_dir)

        self.labels_csv = osp.join(self.features_dir, self.split_group+"Labels.csv")
        self.bbox_split_dir = osp.join(self.bbox_dir, self.split_group)
        self.lmarks_split_dir = osp.join(self.features_dir, "LMarks",  self.split_group)
        self.egaze_split_dir = osp.join(self.features_dir, "EGaze",  self.split_group)
        assert osp.exists(self.labels_csv)
        assert osp.exists(self.bbox_split_dir)
        assert osp.exists(self.lmarks_split_dir)
        assert osp.exists(self.egaze_split_dir)

class FrameData(BasicDataset):

    def __init__(self, frame_num, frame_mat, video_df, bbox_dir, bbox_df, lmarks_df, egaze_df, dataset_dir):

        self.split_group = video_df["Group"]
        BasicDataset.__init__(self, dataset_dir, self.split_group)
        self.frame_num = frame_num
        self.frame = self.cvt_cv_to_pil(frame_mat)
        self.video_df = video_df
        self.bbox_dir = bbox_dir
        self.bbox_df = bbox_df
        self.lmarks_df = lmarks_df  # FIXME: dataframe or series???
        self.egaze_df = egaze_df
        normalize = transforms.Normalize(mean=[127.5, 127.5, 127.5],
                                         std=[127.5, 127.5, 127.5])
        # self.frame_transforms = transforms.Compose(
        #     [transforms.Pad([0, 80]), transforms.Resize([224, 224]), normalize, transforms.ToTensor()])
        # self.bbox_transforms = transforms.Compose(
        #     [transforms.Resize([224, 224]), normalize, transforms.ToTensor()])

        self.transforms_non_delta = transforms.Compose([transforms.Resize([170, 170]), transforms.ToTensor()])
        self.transforms_delta = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

        # these are features
        self.frame_224 = self.transforms_delta(self.frame)
        self.frame_170 = self.transforms_non_delta(self.frame)
        self.bbox_img = self.load_bbox_img()
        self.bbox_224 = self.transforms_delta(self.bbox_img)
        self.bbox_170 = self.transforms_non_delta(self.bbox_img)
        self.bbox_loc = self.scale_bbox_df()
        self.lmarks = self.scale_lmarks()
        self.egaze = self.scale_egaze()
        
    def load_bbox_img(self):
        
        bbox_path = osp.join(self.bbox_split_dir, osp.splitext(self.video_df["ClipID"])[0])
        assert osp.exists(bbox_path), print(bbox_path)
        bbox_path = osp.join(bbox_path, str(self.frame_num)+'.jpg')
        if not osp.exists(bbox_path):
            return Image.new('RGB', (640, 480), color='black')
        return self.load_pil_img(bbox_path)

    def load_pil_img(self, path):
         with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')     
        

    def cvt_cv_to_pil(self, frame_mat):

        self.frame = cv2.cvtColor(frame_mat, cv2.COLOR_BGR2RGB)
        return Image.fromarray(self.frame)

    def scale_lmarks(self):
        width = self.bbox_df["width"].item()
        # return self.lmarks_df[useful_lmarks].divide(width)
        # return {i: self.lmarks_df[i]/width for i in useful_lmarks}
        if width == 0:
            scaled_values = [0 for i in self.useful_lmarks]
        else:
            scaled_values = [self.lmarks_df[i] / width for i in self.useful_lmarks]

        return torch.FloatTensor(scaled_values)

    def scale_egaze(self):
        # return {"x1": self.egaze_df["x1"]/640, "x2": self.egaze_df["x2"]/640}
        return torch.FloatTensor([self.egaze_df["x1"]/640, self.egaze_df["x2"]/640])

    def scale_bbox_df(self):
        # return {
        #     "x": self.bbox_df["x"]/640,
        #     "y": self.bbox_df["y"]/640,
        #     "width": self.bbox_df["width"]/640,
        #     "angle": self.bbox_df["angle"]/180
        # }
        return torch.FloatTensor([self.bbox_df["x"].item()/640, self.bbox_df["y"].item()/640, self.bbox_df["width"].item()/640, self.bbox_df["angle"].item()/180])


class VideoData(BasicDataset):

    def __init__(self, video_df, dataset_dir, subsampled_indices):

        self.split_group = video_df["Group"]
        BasicDataset.__init__(self, dataset_dir, self.split_group)
        self.subsampled_indices = subsampled_indices
        this_video_bbox_dir = osp.join(self.bbox_split_dir, osp.splitext(video_df["ClipID"])[0])
        this_video_bbox_df = pd.read_csv(osp.join(this_video_bbox_dir, "box.csv"))
        this_video_lmarks_df = pd.read_csv(
            osp.join(self.lmarks_split_dir, osp.splitext(video_df["ClipID"])[0]+".csv"))
        this_video_egaze_df = pd.read_csv(
            osp.join(self.egaze_split_dir, osp.splitext(video_df["ClipID"])[0]+".csv"))
        assert osp.exists(video_df["Path"]), video_df["Path"]
        assert osp.exists(this_video_bbox_dir)
        assert len(this_video_bbox_df) != 0
        assert len(this_video_lmarks_df) != 0
        assert len(this_video_egaze_df) != 0

        cap = cv2.VideoCapture(video_df["Path"])

        self.spatial_info = {}
        for i in range(1, 301):
            ret, frame = cap.read()
            #print(frame.shape)
            if not ret:
                frame = np.zeros((480, 640, 3), np.uint8)
            if i in self.subsampled_indices:
                bbox_info = this_video_bbox_df[this_video_bbox_df.frame == i]
                if len(bbox_info) == 0:
                    bbox_info = pd.DataFrame({'frame': i,
                                              'x': 0,
                                              'y': 0,
                                              'width': 0,
                                              'angle': 0,
                                              'score': 0
                                              }, columns=['frame','x','y','width','angle','score'], index=[0])
                lmarks_info = this_video_lmarks_df.iloc[i-1]
                egaze_info = this_video_egaze_df.iloc[i-1]
                
                self.spatial_info[i] = FrameData(i,
                                                 frame,
                                                 video_df,
                                                 this_video_bbox_dir,
                                                 bbox_info,
                                                 lmarks_info,
                                                 egaze_info,
                                                 dataset_dir)

        cap.release()
        self.temporal_info = {
            subsampled_indices[j]: self.make_diff(
                self.spatial_info,
                subsampled_indices[j-1],
                subsampled_indices[j]) for j in range(1, len(subsampled_indices))
        }

        self.temporal_info[subsampled_indices[0]] = copy.deepcopy(
            self.temporal_info[subsampled_indices[1]])
        

    def make_diff(self, spatial_info, i, j):
        t0 = spatial_info[i]
        t1 = spatial_info[j]
        temporal_dict = {
            "frame": t1.frame_224 - t0.frame_224,
            "bbox_img": 100 * (t1.bbox_224 - t0.bbox_224),
            "bbox_loc": 100 * (t1.bbox_loc - t0.bbox_loc),
            "lmarks": 100 * (t1.lmarks - t0.lmarks),
            "egaze": 100 * (t1.egaze - t0.egaze)
        }
        return temporal_dict


class DAiSEEDataset(BasicDataset):

    def __init__(self, dataset_dir, affection, split_group, subsample_rate=3):

        BasicDataset.__init__(self, dataset_dir, split_group)
        self.subsample_rate = subsample_rate
        self.split_group = split_group
        self.affection = affection

        assert self.split_group in ["Test", "Train", "Validation"]
        assert self.affection in ["Engagement", "Boredom", "Confusion", "Frustration"]
        assert osp.exists(self.labels_csv)

        self.labels_df = pd.read_csv(self.labels_csv)
        self.subsampled_indices = self.subsampling_rate2_indices(
            self.subsample_rate, start_index=1, n_frames=300)

    def subsampling_rate2_indices(self, rate, start_index=1, n_frames=300):

        return [i for i in range(start_index, n_frames+1, rate)]

    def one_hot(self, i):
        i = int(i)
        if i == 0:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        elif i == 1:
            return np.array([0, 1, 0, 0], dtype=np.float32)
        elif i == 2:
            return np.array([0, 0, 1, 0], dtype=np.float32)
        elif i == 3:
            return np.array([0, 0, 0, 1], dtype=np.float32)
        else:
            raise Exception("Could not convert to one_hot encoding")

    def __len__(self):

        return len(self.labels_df)

    def __getitem__(self, index):

        selection = self.labels_df.iloc[index]
        video_info = VideoData(selection, self.base_dir, self.subsampled_indices)
        label = self.one_hot(selection[self.affection])
        spatial_info = {i: [video_info.spatial_info[i].bbox_170, video_info.spatial_info[i].lmarks, video_info.spatial_info[i].egaze] for i in self.subsampled_indices}
        temporal_info = video_info.temporal_info
        del video_info
        
        return [(spatial_info, temporal_info), label]
