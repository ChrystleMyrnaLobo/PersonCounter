from imutils.video import VideoStream
import argparse
import imutils
import cv2 # 3.4.5
from utils.dataset import MOT16

import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger()

class TrackerPersonCounter:
    # Traker from OpenCV
    def __init__(self, vid, tkr, pfr, dfr):
        logger.info("Running tracker based person counter")
        self.path_to_video = vid
        self.tracker_algo = tkr
        self.ds = MOT16(10)
        self.process_frame_rate = pfr
        self.detect_frame_rate = dfr

    def str(self):
        txt = "Tracker person counter \n"
        txt += self.path_to_video + "\n"
        txt += self.tracker_algo + "\n"
        return txt

    def setup(self):
        self.video_stream = cv2.VideoCapture(self.path_to_video) # open video
        self.gt_ds = self.ds.parseGroundtruth(targetDS="OpenCV") # Extract BB from GT

    def initializeFrame(self, frame, frame_id = 1):
        logger.info("Reset trackers " + str(frame_id))
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        # KCF: Fast and accurate
        # CSRT: More accurate than KCF but slower
        # MOSSE: Extremely fast but not as accurate as either KCF or CSRT
        # create a new object tracker for the bounding box and add it to our multi-object tracker
        image_id = str(frame_id).zfill(6)
        cnt = self.gt_ds[image_id]['num_detections']
        bbs = self.gt_ds[image_id]['groundtruth_boxes']
        self.trackers = cv2.MultiTracker_create() # Multi object tracker
        for bb in bbs:
            box = tuple(bb)
            object_tracker = OPENCV_OBJECT_TRACKERS[self.tracker_algo]()
            self.trackers.add(object_tracker, frame, box)

    def run(self):
        frame_id = 1
        frame = self.video_stream.read()[1]
        self.initializeFrame(frame, frame_id)

        while True:
            frame = self.video_stream.read()[1]
            if frame is None: # End of video
                break
            # Processing rate of system
            if frame_id % self.process_frame_rate != 0:
                frame_id += 1
                continue
            # Detect frame rate
            if (frame_id/self.process_frame_rate + 1) % self.detect_frame_rate == 0:
                frame_id += 1
                self.initializeFrame(frame, frame_id) # Perform only detection
                continue

            frame_id += 1
            (success, boxes) = self.trackers.update(frame) # compute the update to tracker

            # draw BB on the frame
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        self.video_stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="path to input video file")
    parser.add_argument("-t", "--tracker", type=str, default="kcf",	help="OpenCV object tracker type")
    parser.add_argument("-pfr", "--processfr", type=int, default="5",	help="Processing frame rate")
    parser.add_argument("-dfr", "--detectfr", type=int, default="10",	help="detect frame rate")
    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(str(key) + ': ' + str(value))

    tpc = TrackerPersonCounter(args.video, args.tracker, args.processfr, args.detectfr)
    tpc.setup()
    tpc.run()
    logger.info("Done")
