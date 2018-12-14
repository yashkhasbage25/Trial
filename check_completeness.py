import os
import time
import argparse

import pandas as pd
import os.path as osp


def get_parsed_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feat", type=str, required=True, help="Path to features dir")
    args = parser.parse_args()
    return args


class Checker:

    def __init__(self, features_dir):

        self.features_dir = features_dir
        assert osp.exists(self.features_dir)

        self.splits = ["Test", "Train", "Validation"]

        self.labels_csv = {}
        self.labels_df = {}
        self.bbox_dir = osp.join(self.features_dir, "BBox")
        self.lmarks_dir = osp.join(self.features_dir, "LMarks")
        self.egaze_dir = osp.join(self.features_dir, "EGaze")

        assert osp.exists(self.bbox_dir)
        assert osp.exists(self.lmarks_dir)
        assert osp.exists(self.egaze_dir)

        self.log_dir = "CompletenessCheck"
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.bbox_split_dir = {}
        self.lmarks_split_dir = {}
        self.egaze_split_dir = {}

        for split in self.splits:

            self.labels_csv[split] = osp.join(
                self.features_dir, split + "Labels.csv")
            self.bbox_split_dir[split] = osp.join(self.bbox_dir, split)
            self.lmarks_split_dir[split] = osp.join(self.lmarks_dir, split)
            self.egaze_split_dir[split] = osp.join(self.egaze_dir, split)

            assert osp.exists(self.labels_csv[split])
            assert osp.exists(self.bbox_split_dir[split])
            assert osp.exists(self.lmarks_split_dir[split])
            assert osp.exists(self.egaze_split_dir[split])

            self.labels_df[split] = pd.read_csv(self.labels_csv[split])


class BBoxChecker(Checker):

    def __init__(self, features_dir):

        Checker.__init__(self, features_dir)
        self.features_dir = features_dir
        self.f = open(
            osp.join(self.log_dir, "BBoxChecker.txt"), mode='w+')

        for split in self.splits:
            self.check_split_groupwise(split)
        self.f.close()

    def check_split_groupwise(self, split):
        for _, row in self.labels_df[split].iterrows():
            bbox_dir = osp.join(self.bbox_split_dir[split], osp.splitext(row["ClipID"])[0])
            if not osp.exists(bbox_dir):
                self.f.write("File not found: %s\n" % bbox_dir)
            else:
                for i in range(1, 301):
                    img_path = osp.join(bbox_dir, "%d.jpg" % i)
                    if not osp.exists(img_path):
                        self.f.write("Img not found: %s\n" % img_path)
                bbox_csv = osp.join(bbox_dir, "box.csv")
                if not osp.exists(bbox_csv):
                    self.f.write("File not found: %s\n" % bbox_csv)
                else:
                    bbox_df = pd.read_csv(bbox_csv)
                    if bbox_df.shape != (300, 6):
                        self.f.write("File incomplete: %s\n" % bbox_csv)


class LMarksChecker(Checker):

    def __init__(self, features_dir):

        Checker.__init__(self, features_dir)
        self.features_dir = features_dir
        self.f = open(osp.join(self.log_dir, "LMarksChecker.txt"), mode='w+')

        for split in self.splits:
            self.check_split_groupwise(split)
        self.f.close()

    def check_split_groupwise(self, split):
        for _, row in self.labels_df[split].iterrows():
            lmarks_csv = osp.join(
                self.lmarks_split_dir[split], osp.splitext(row["ClipID"])[0]+".csv")
            if not osp.exists(lmarks_csv):
                self.f.write("File not found: %s\n" % lmarks_csv)
            else:
                lmarks_df = pd.read_csv(lmarks_csv)
                if lmarks_df.shape != (300, 138):
                    self.f.write("File incomplete: %s\n" % lmarks_csv)


class EGazeChecker(Checker):

    def __init__(self, features_dir):

        Checker.__init__(self, features_dir)
        self.features_dir = features_dir
        self.f = open(osp.join(self.log_dir, "EGazeChecker.txt"), mode='w+')

        for split in self.splits:
            self.check_split_groupwise(split)
        self.f.close()

    def check_split_groupwise(self, split):
        for _, row in self.labels_df[split].iterrows():
            egaze_csv = osp.join(self.egaze_split_dir[split], osp.splitext(row["ClipID"])[0]+".csv")
            if not osp.exists(egaze_csv):
                self.f.write("File not found: %s\n" % egaze_csv)
            else:
                egaze_df = pd.read_csv(egaze_csv)
                if egaze_df.shape != (300, 2):
                    self.f.write("File incomplete: %s\n" % egaze_csv)


class AllChecker:

    def __init__(self, features_dir):

        print("Started BBox")
        start = time.time()
        _ = BBoxChecker(features_dir)
        end = time.time()
        print("Elapsed time: {:2f}".format(end-start))
        print("Started LMarks")
        start = time.time()
        _ = LMarksChecker(features_dir)
        end = time.time()
        print("Elapsed time: {:2f}".format(end-start))
        print("Started Egaze")
        start = time.time()
        _ = EGazeChecker(features_dir)
        end = time.time()
        print("Elapsed time: {:2f}".format(end-start))


if __name__ == '__main__':

    args = get_parsed_args()
    checker = AllChecker(args.feat)
