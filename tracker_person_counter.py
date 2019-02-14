# from imutils.video import VideoStream
import argparse
import imutils
import cv2 # 3.4.5
from utils.dataset import MOT16
from utils.pc_utils import pc_PerImageEvaluation

import time, os, math
import pandas
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger()

class TrackerPersonCounter:
    """
    Detect and track approach for person counter using OpenCV object tracking algorithms.
    Tracker is initialized with bounding box from detection (or ground truth) and following frames are tracked.
    The tracker is reinitialized at a frame rate of of <detectfr>
    """
    def __init__(self, ds, tkr, dt, ws):
        logger.info("Running tracker based person counter")
        self.ds = ds
        self.tracker_algo = tkr
        # frames skipped while detection = detection speed * video frame rate
        self.detect_lag = math.floor(dt * ds.frame_rate)
        self.window_size = ws
        filename =  ds.video_name + "_" + self.tracker_algo + "_pfr" + str(dt) + "_ws" + str(self.window_size) + '.csv'
        logger.info("\nFilename {}\n".format(filename))
        self.path_to_output_file = os.path.join("output", filename)

    def str(self):
        txt = "Tracker person counter \n"
        txt += self.path_to_video + "\n"
        txt += self.tracker_algo + "\n"
        return txt

    def setup(self):
        #TODO pick between video and directory
        self.video_stream = cv2.VideoCapture(self.ds.getVideoStream()) #TODO resize
        # self.video_stream = cv2.VideoCapture(self.path_to_video) # open video
        self.gt_ds = self.ds.parseGroundtruth(targetDS="OpenCV") # Extract BB from GT in OpenCV tracker format
        self.result = [] # Result per BB

    def detectOnFrame(self, frame, frame_id = 1):
        # Run detection, and filter out based on category and confidence
        image_id = str(frame_id).zfill(6)
        bbs = self.gt_ds[image_id]['groundtruth_boxes'] # Using saved inference

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

        self.trackers = cv2.MultiTracker_create() # Multi object tracker
        #TODO corelate
        # create a new object tracker for each object and add it to our multi-object tracker
        for bb in bbs:
            box = tuple(bb)
            object_tracker = OPENCV_OBJECT_TRACKERS[self.tracker_algo]()
            self.trackers.add(object_tracker, frame, box)
        self.log_output(bbs, frame_id, "detect", self.detect_lag)

    def trackOnFrame(self, frame, frame_id):
        start = time.time()
        (success, bbs) = self.trackers.update(frame) # update tracker
        # frames skipped while tracking = tracking speed * video frame rate
        self.track_lag = math.floor( (time.time() - start) * self.ds.frame_rate )
        self.log_output(bbs, frame_id, "track", self.track_lag)

    def log_output(self, bbs, frame_id, phase, lag):
        """ Log the output for each frame, either detect or track """
        person_id = 1
        for box in bbs:
            (x, y, w, h) = [int(v) for v in box]
            row = [frame_id, phase, person_id, x, y, w, h, lag]
            person_id += 1
            self.result.append(row)
        pass

    def run(self):
        """ Detect on leader frame, skipping frames received during the detection; Track subsequent k (window size) frames, skipping frames received during the tracking """
        frame_id = 1 # Received frame
        next_frame_id = 1 # Next frame to process
        k = self.window_size

        while True:
            frame = self.video_stream.read()[1]
            if frame is None: # End of video
                break
            # Skip frames received during processing
            if frame_id != next_frame_id:
                frame_id += 1
                continue
            # Detect OR track
            if k == self.window_size: # Window completed
                self.detectOnFrame(frame, frame_id)
                next_frame_id += self.detect_lag
                k = 0
                logger.info("Frame {} DETECT. Skipping {} frame(s) to {}".format(frame_id, self.detect_lag, next_frame_id) )
            else: # Track
                self.trackOnFrame(frame, frame_id)
                next_frame_id += self.track_lag
                k += 1
                logger.info("Frame {} TRACK {}. Skipping {} frame(s) to {}".format(frame_id, k, self.track_lag, next_frame_id) )
            frame_id += 1

        logger.info("Save log to {}".format(self.path_to_output_file))
        pd = pandas.DataFrame(self.result)
        pd.to_csv(self.path_to_output_file, header=False) # contains index

if __name__ == '__main__' :
    # python tracker_person_counter.py -v MOT16-10 -dh ~/4Sem/MTP1/MOT16
    # 2>&1 | tee output/log.txt # Stream log to file
    parser = argparse.ArgumentParser()
    parser.add_argument("-dh", "--dataset_home", type=str, required=True, help="path to dataset home")
    parser.add_argument("-v", "--video", type=str, required=True, help="video stream. e.g: MOT16-10")
    parser.add_argument("-t", "--tracker", type=str, default="kcf",	help="OpenCV object tracker type. Pick from kcf, csrt, mosse, boosting, mil, tld, medianflow ")
    parser.add_argument("-dt", "--detect_speed", type=float, default="0.5",	help="detection speed (sec)")
    parser.add_argument("-w", "--window_size", type=int, default="10",	help="Window size (#frames) of tracking")

    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(str(key) + ': ' + str(value))

    # Dataset
    dataset_name, vid = args.video.split('-')
    if dataset_name == "MOT16":
        ds = MOT16(args.dataset_home, int(vid))
    else:
        logger.error("Invalid dataset")
        return
    tpc = TrackerPersonCounter(ds, args.tracker, args.detect_speed, args.window_size)
    tpc.setup()
    tpc.run()

    logger.info("Done")