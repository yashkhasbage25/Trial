import sys
import glob
import os
import argparse

import pandas as pd
import os.path as osp
import shutil as sh


def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--orig", type=str, help="path to original dataset directory")
    parser.add_argument("-d", "--dup", type=str, help="path to duplicate directory")
    parser.add_argument("-n", "--num", type=int, default=10, help="total videos in dataset")

    return parser


class Dataset:

    def __init__(self, root_path):

        self.root_dir = root_path

        assert osp.exists(self.root_dir), "Main directory not found"
        self.dataset_dir = osp.join(self.root_dir, "DataSet")
        self.features_dir = osp.join(self.root_dir, "Features")

        self.splits = ["Test", "Train", "Validation"]
        self.split_csv = {}
        self.split_df = {}

        self.get_labels()

        self.bbox_dir = osp.join(self.features_dir, "BBox")
        self.lmarks_dir = osp.join(self.features_dir, "LMarks")
        self.egaze_dir = osp.join(self.features_dir, "EGaze")

        self.bbox_split_dirs = {split: osp.join(self.bbox_dir, split) for split in self.splits}
        self.lmakrs_split_dirs = {split: osp.join(self.lmarks_dir, split) for split in self.splits}
        self.egaze_split_dirs = {split: osp.join(self.egaze_dir, split) for split in self.splits}

        self.make_dirs([self.features_dir, self.dataset_dir,
                        self.bbox_dir, self.lmarks_dir, self.egaze_dir])

        self.create_partitions([self.dataset_dir, self.bbox_dir, self.lmarks_dir, self.egaze_dir])

    def make_dirs(self, path_list):

        for path in path_list:
            if not osp.exists(path):
                os.mkdir(path)

    def get_labels(self):  # TODO: remove for trial dataset

        for split in self.splits:
            self.split_csv[split] = osp.join(self.features_dir, split+"Labels.csv")
            if osp.exists(self.split_csv[split]):
                self.split_df[split] = pd.read_csv(self.split_csv[split])
            else:
                self.split_df[split] = None
        self.split_csv["All"] = osp.join(self.features_dir, "AllLabels.csv")
        if osp.exists(self.split_csv["All"]):
            self.split_df["All"] = pd.read_csv(
                osp.join(self.features_dir, "AllLabels.csv"))
        else:
            self.split_df["All"] = None

    def create_partitions(self, dirs):
        for dir in dirs:
            split_dirs = [osp.join(dir, split) for split in self.splits]
            for subdir in split_dirs:
                if not osp.exists(subdir):
                    os.mkdir(subdir)


class DataSampler:

    def __init__(self, df, dataset_dir):

        self.df = df.sample(frac=1)
        self.dataset_len = len(df)
        self.splits = ["Test", "Train", "Validation"]
        self.dataset_dir = dataset_dir

        self.train_len = int(self.dataset_len*0.6)
        self.validation_len = int(self.dataset_len*0.2)
        self.test_len = int(self.dataset_len*0.2)
        assert self.train_len + self.validation_len + self.test_len == self.dataset_len

        self.split_history = self.df[["ClipID", "Group"]]

        self.df = self.split_data()
        self.df = self.rectify_group_labels()

    def split_data(self):

        train_df = self.df.iloc[:self.train_len]
        test_df = self.df.iloc[self.train_len:self.train_len+self.test_len]
        validation_df = self.df.iloc[self.train_len+self.test_len:]

        assert len(train_df) == self.train_len
        assert len(test_df) == self.test_len
        assert len(validation_df) == self.validation_len

        df_dict = {"Test": test_df, "Train": train_df, "Validation": validation_df}
        return df_dict

    def rectify_group_labels(self):

        df_dict = {}
        for split in self.splits:
            df_dict[split] = pd.DataFrame(
                columns=["ClipID", "Path", "Group", "Boredom", "Engagement", "Confusion", "Frustration"])
            for _, row in self.df[split].iterrows():
                video_name = row.at["ClipID"]
                new_path = osp.join(self.dataset_dir, split, video_name)
                sh.copy(row["Path"], new_path)
                df_dict[split] = df_dict[split].append({
                    "ClipID": row["ClipID"],
                    "Path": new_path,
                    "Group": split,
                    "Boredom": row["Boredom"],
                    "Engagement": row["Engagement"],
                    "Confusion": row["Confusion"],
                    "Frustration": row["Frustration"]
                }, ignore_index=True)
        return df_dict

    def get_split_history(self):

        return self.split_history


class DuplicateMaker:

    def __init__(self, original_dataset, trial_dataset, dataset_len=100):

        self.orig = Dataset(original_dataset)
        self.trial = Dataset(trial_dataset)
        self.dataset_len = dataset_len

        self.splits = ["Test", "Train", "Validation"]
        self.duplicate_dataset()

    def duplicate_dataset(self):

        sampled_videos = DataSampler(self.orig.split_df["All"].sample(
            n=self.dataset_len), dataset_dir=self.trial.dataset_dir)
        self.trial.split_df = sampled_videos.df

        self.split_history = sampled_videos.get_split_history()
        self.save_label_df()
        self.duplicate_bbox()
        self.duplicate_lmarks()
        self.duplicate_egaze()

    def save_label_df(self):

        for split in self.splits:
            self.trial.split_df[split].to_csv(self.trial.split_csv[split], mode='w+', index=False)

    def duplicate_bbox(self):

        for split in self.splits:
            for _, row in self.trial.split_df[split].iterrows():
                clip_id = row["ClipID"]
                prev_split = self.split_history[self.split_history.ClipID ==
                                                row["ClipID"]]["Group"].item()
                sh.copytree(
                    osp.join(self.orig.bbox_split_dirs[prev_split],
                             osp.splitext(clip_id)[0]),
                    osp.join(self.trial.bbox_split_dirs[split], osp.splitext(clip_id)[0]))

    def duplicate_lmarks(self):

        for split in self.splits:
            for _, row in self.trial.split_df[split].iterrows():
                clip_id = row["ClipID"]
                prev_split = self.split_history[self.split_history.ClipID ==
                                                row["ClipID"]]["Group"].item()
                sh.copy(
                    osp.join(self.orig.lmarks_split_dirs[prev_split],
                             osp.splitext(clip_id)[0]+".csv"), osp.join(self.trial.lmarks_split_dirs[split], clip_id))

    def duplicate_egaze(self):

        for split in self.splits:
            for _, row in self.trial.split_df[split].iterrows():
                clip_id = row["ClipID"]
                prev_split = self.split_history[self.split_history.ClipID ==
                                                row["ClipID"]]["Group"].item()
                sh.copy(
                    osp.join(self.orig.egaze_split_dirs[prev_split],
                             osp.splitext(clip_id)[0]+".csv"), osp.join(self.trial.egaze_split_dirs[split], clip_id))


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    duplicator = DuplicateMaker(original_dataset=args.orig,
                                trial_dataset=args.dup, dataset_len=10)
