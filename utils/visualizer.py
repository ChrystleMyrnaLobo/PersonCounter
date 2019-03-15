import argparse
import imutils
import cv2 # 3.4.5
from utils.dataset import MOT16

import time, os, math
import pandas as pd

class Visualizer:
    """ Utils to show the BB on the image """
    def __init__(self, ds):
        self.ds = ds

    def str(self):
        txt = "Visualizer \n"
        return txt

    def setup(self):
        """ Initialized video stream and detection source"""
        self.video_stream = cv2.VideoCapture(self.ds.getVideoStream())
        self.gt_df = self.ds.parseAnnotation_OpenCV() #  GT BB in OpenCV tracker format as Pandas DataFrame

    def showInference(self, frame, boxes, frame_id, phase):
        """ Display the image and BB """
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame {} {}".format(phase, frame_id), frame)
        key = cv2.waitKey(2000)
        # if key == 27: # Esc key
            # return
        cv2.destroyAllWindows()

    def perFrame(self,frame_id, frame=None):
        """ Per frame """
        if frame == None:
            # read from directory
            frame = cv2.imread(self.ds.getFramePath(frame_id), cv2.IMREAD_COLOR)
        bbs = self.gt_df.loc[self.gt_df.frame_id == frame_id][['xmin', 'ymin', 'width', 'height']].values
        self.showInference(frame, bbs, frame_id, "")

    def allFrames(self):
        frame = self.video_stream.read()[1]
        for frame_id in self.gt_df.frame_id.unique():
            self.perFrame(frame_id)
        pass

    def filterFrames(self, person_id):
        """ Pick frames having person of interest """
        print("Original count {}".format(self.gt_df.shape[0]))
        print(self.gt_df.person_id.unique())
        self.gt_df = self.gt_df.loc[self.gt_df.person_id == person_id]
        # self.gt_df = self.gt_df.loc[self.gt_df.frame_id >= 350]
        print("After filtering person {} count {}".format(person_id, self.gt_df.shape[0]))
        pass

    def custom(self, frame_id):
        """ Draw custom BB """
        frame = cv2.imread(self.ds.getFramePath(frame_id), cv2.IMREAD_COLOR)
        bbs = [[842,350,46,59]]
        self.showInference(frame, bbs, frame_id, "")

if __name__ == '__main__' :
    # 2>&1 | tee output/log.txt # Stream log to file
    parser = argparse.ArgumentParser()
    parser.add_argument("-dh", "--dataset_home", type=str, default="/home/chrystle/4Sem/MTP1/MOT16", help="path to dataset home")
    parser.add_argument("-v", "--video", type=str, default="MOT16-10", help="video stream. e.g: MOT16-10")

    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        print(str(key) + ': ' + str(value))

    # Dataset
    dataset_name, vid = args.video.split('-')
    if dataset_name == "MOT16":
        ds = MOT16(args.dataset_home, int(vid))
    else:
        logger.error("Invalid dataset")
        exit()

    vzr = Visualizer(ds)
    vzr.setup()
    # vzr.custom(366)
    vzr.filterFrames(45)
    # vzr.perFrame(354)
    vzr.allFrames()
