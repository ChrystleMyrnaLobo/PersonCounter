import os
import csv
import numpy as np
from utils.od_utils import *
import configparser

class MOT16:
    """ Notes on the MOT16 dataset:
        gt.tx is CSV text-file containing one object instance per line. Each line must contain 10 values:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence score>, <class id>, <visibility>
        Top 3 records for video 10
        1,1,1368,394,74,226,1,1,1
        2,1,1366,394,75,229,1,1,1
        3,1,1365,394,76,232,1,1,1
        All frame numbers, target IDs and bounding boxes are 1-based.
        Frame number is image number without leading 0. Person ID is <id>
        confidence is flag indicating if the entry is to be considered (1) or ignored (0)
        class id is type of object annotated - pedestrian, person on vehicle, car, motorbike, static person
        visibility how much of that object is visible (0 to 1) due to occlusion or going out of frame

        Directory structure of MOT16
        MOT16/
         train/
            MOT16-xx/
                det/det.txt
                gt/gt.txt
                img1.*.jpg
                seqinfo.ini
    """
    def __init__(self, home, video_id):
        self.dataset_name = "MOT16"
        self.path_to_home = home #
        self.video_name = 'MOT16-' + str(video_id).zfill(2)
        # Set path
        if video_id in [2,4,5,9,10,11,13]: # Train data
            self.path_to_dataset_dir = os.path.join( self.path_to_home, 'train', self.video_name)
        else:
            self.path_to_dataset_dir = os.path.join( self.path_to_home, 'test', self.video_name)
        # self.image_count = self.getImageCount()
        self.parseINI()
        self.path_to_image_dir = os.path.join(self.path_to_dataset_dir, 'img1')
        self.path_to_annotation_dir = os.path.join(self.path_to_dataset_dir, 'gt', 'gt.txt')
        # Dataset categories
        self.od_dir = os.path.join(os.pardir, 'obj_det')
        self.num_classes = 12
        self.category_index = load_category_index('mot16_label_map.pbtxt', self.num_classes) # Dataset has these categories

    def getVideoStream(self):
        """ Image directory as "MOT16/train/MOT16-10/img1/%6d.jpg" """
        return self.path_to_image_dir + "/%6d.jpg"

    def str(self):
        txt = "class MOT16\n"
        txt += self.path_to_home + "\n"
        txt += self.path_to_dataset_dir + "\n"
        txt += self.path_to_annotation_dir + "\n"
        txt += self.path_to_image_dir + "\n"
        txt += str(self.image_count)
        return txt

    def parseINI(self):
        # Read seqinfo.ini
        path_ini_file = os.path.join(self.path_to_dataset_dir, 'seqinfo.ini')
        parser = configparser.ConfigParser()
        parser.read(path_ini_file)
        self.image_count = int( parser.get('Sequence', 'seqLength') )
        self.frame_rate = int( parser.get('Sequence', 'frameRate') )

    def mot_to_od(self, row):
        """ Convert MOT16 record format to OD api format"""
        [frame_id, person_id, xmin, ymin, width, height, conf, class_idx, visibility] = row
        gt_bb = []
        # Extract as per MOT16
        ymax = float(ymin) + float(height) # ymax # bb_top + bb_height
        xmax = float(xmin) + float(width) # xmax # bb_left + bb_width
        # Insert as per OD
        gt_bb = [ymin, xmin, ymax, xmax]
        gt_bb = [float(i) for i in gt_bb] #map(float, gt_bb)
        return int(frame_id), int(person_id), gt_bb, bool(int(conf)), int(class_idx)

    def mot_to_opencv(self, row):
        """ Convert MOT16 record format to OpenCV tracker api format"""
        [frame_id, person_id, xmin, ymin, width, height, conf, class_idx, visibility] = row
        # Need to scale
        # video w x h = 960 x 540
        # image w x h = 1920 x 1080
        scale_x = 1 #960.0 / 1920.0
        scale_y = 1 #540.0 / 1080.0
        gt_bb = [scale_x * float(xmin), scale_y * float(ymin), scale_x * float(width), scale_y * float(height)]
        # gt_bb = [float(i) for i in gt_bb]
        return int(frame_id), int(person_id), gt_bb, bool(int(conf)), int(class_idx)

    def bb_od_to_mot(self, detection_box):
        """ Convert BB from OD format to MOT16 format"""
        row = []
        [ymin, xmin, ymax, xmax] = detection_box
        width = float(xmax) - float(xmin) # bb_width # x_max - xmin
        height = float(ymax) - float(ymin) # bb_height # ymax - ymin
        row = [xmin, ymin, width, height]
        row = map(float, row)
        return row

    def parseGroundtruth(self, asDetection=False, targetDS="OD"): #extractGT
        """" Read the groundtruth into a dictionary as per OD API format. Return as dt_dict if asDetection is true """
        gt_ds = {} # groundtruth for all images in dataset
        #person_cnt, person_id = 0, 1
        with open(self.path_to_annotation_dir, 'r') as csvfile:
            #print "Reading ground groundtruth from " + self.path_to_annotation_dir + " as detection " + str(asDetection)
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                gt_dict = {}
                if targetDS == "OpenCV":
                    frame_id, person_id, gt_bb, conf, class_idx = self.mot_to_opencv(row)
                else:
                    frame_id, person_id, gt_bb, conf, class_idx = self.mot_to_od(row)
                #Ignore cases
                if not conf or frame_id > self.image_count:
                    # Read only till image_count frames
                    continue
                image_id = str(frame_id).zfill(6)
                if image_id in gt_ds :
                    gt_dict = gt_ds[image_id]
                    gt_dict['num_detections'] += 1
                    gt_dict['person_id'] = np.append( gt_dict['person_id'], [ person_id ] )
                    if asDetection:
                        #class_idx = 1 # MSCOCO has only one class for this dataset i.e. Person
                        gt_dict['detection_boxes'] = np.append( gt_dict['detection_boxes'], [gt_bb], axis=0)
                        gt_dict['detection_classes'] = np.append( gt_dict['detection_classes'], [class_idx])
                        gt_dict['detection_scores'] = np.append( gt_dict['detection_scores'], [1])
                    else :
                        gt_dict['groundtruth_boxes'] = np.append( gt_dict['groundtruth_boxes'], [gt_bb], axis=0 )
                        gt_dict['groundtruth_classes'] = np.append( gt_dict['groundtruth_classes'], [class_idx] )
                        gt_dict['groundtruth_scores'] = np.append( gt_dict['groundtruth_scores'], [1])
                else:
                    gt_dict['num_detections'] = 1
                    gt_dict['person_id'] = np.array( [ person_id ] )
                    if asDetection:
                        #class_idx = 1 # MCCOCO has only one class for this dataset i.e. Person
                        gt_dict['detection_boxes'] = np.array( [gt_bb], dtype="float32")
                        gt_dict['detection_classes'] = np.array( [class_idx] )
                        gt_dict['detection_scores'] = np.array([1], dtype="int")
                    else:
                        gt_dict['groundtruth_boxes'] = np.array( [gt_bb] , dtype="float32")
                        gt_dict['groundtruth_classes'] = np.array( [class_idx] )
                        gt_dict['groundtruth_scores'] = np.array([1])
                gt_ds[image_id] = gt_dict
        return gt_ds

    def parseDetection(self, image_id, dt_dict):
        """ Convert detection dict into csv rows as per MOT16 format"""
        rows = []
        for i in range(0,dt_dict['num_detections']):
            row = []
            row.append(image_id) # frame id
            row.append(dt_dict['detection_classes'][i]) # ID
            gt_bb = self.bb_od_to_mot(dt_dict['detection_boxes'][i])
            row.extend( gt_bb )
            rows.append(row)
        return rows
