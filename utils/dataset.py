import os
import csv
import numpy as np
from object_detection.utils import label_map_util

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
    """
    def __init__(self, video_id):
        self.dataset_name = "MOT16"
        self.video_seq = str(video_id).zfill(2)
        # Set path
        if video_id in [2,4,5,9,10,11,13]: # Train data
            self.path_to_dataset_dir = os.path.join( os.pardir, self.dataset_name, 'train', 'MOT16-' + self.video_seq)
        else:
            self.path_to_dataset_dir = os.path.join( os.pardir, self.dataset_name, 'test', 'MOT16-' + self.video_seq)
        self.image_count = self.getImageCount()
        self.path_to_image_dir = os.path.join(self.path_to_dataset_dir, 'img1')
        self.path_to_annotation_dir = os.path.join(self.path_to_dataset_dir, 'gt', 'gt.txt')
        # Dataset categories
        self.od_dir = os.path.join(os.pardir, 'obj_det')
        self.num_classes = 12
        self.loadCategoryIndex()

    def loadCategoryIndex(self):
        """ Dataset is annotated on these category """
        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join('data', 'mot16_label_map.pbtxt')
        path_to_labels = os.path.join(self.od_dir, path_to_labels)

        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`. Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        #  str(self.category_index[1]['name'])
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def str(self):
        txt = "class MOT16\n"
        txt += self.path_to_dataset_dir + "\n"
        txt += self.path_to_annotation_dir + "\n"
        txt += self.path_to_image_dir + "\n"
        txt += str(self.image_count)
        return txt

    def getImageCount(self):
        # Read seqinfo.ini to get seqLength=654
        path_info_file = os.path.join(self.path_to_dataset_dir, 'seqinfo.ini')
        with open(path_info_file,'rb') as fd:
            meta = fd.read()
            idx = meta.find('seqLength')
            meta[idx+10:idx+13]
        return 3
        #return int( meta[idx+10:idx+13] )

    def mot_to_od(self, row):
        """ Convert MOT16 record format to OD api format"""
        [frame_id, person_id, xmin, ymin, width, height, conf, class_idx, visibility] = row
        gt_bb = []
        # Extract as per MOT16
        #ymin = float(row[3])
        #xmin = float(row[2])
        ymax = float(ymin) + float(height)
        xmax = float(xmin) + float(width)
        # Insert as per OD
        gt_bb.append( ymin ) # ymin # bb_top
        gt_bb.append( xmin ) # xmin # bb_left
        gt_bb.append( ymax ) # ymax # bb_top + bb_height
        gt_bb.append( xmax ) # xmax # bb_left + bb_width
        gt_bb = map(float, gt_bb)
        return int(frame_id), int(person_id), gt_bb, bool(int(conf)), int(class_idx)

    def bb_od_to_mot(self, detection_box):
        """ Convert BB from OD format to MOT16 format"""
        row = []
        [ymin, xmin, ymax, xmax] = detection_box
        width = float(xmax) - float(xmin)
        height = float(ymax) - float(ymin)
        row.append(xmin) # bb_left #xmin
        row.append(ymin) # bb_top # ymin
        row.append(width) # bb_width # x_max - xmin
        row.append(height) # bb_height # ymax - ymin
        row = map(float, row)
        return row

    def parseGroundtruth(self, asDetection=False): #extractGT
        """" Read the groundtruth into a dictionary as per OD API format. Return as dt_dict if asDetection is true """
        gt_ds = {} # groundtruth for all images in dataset
        #person_cnt, person_id = 0, 1
        with open(self.path_to_annotation_dir, 'rb') as csvfile:
            print "Reading ground groundtruth from " + self.path_to_annotation_dir + " as detection " + asDetection
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                gt_dict = {}
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
            row.append(dt_dict['person_id'][i]) # ID
            gt_bb = self.bb_od_to_mot(dt_dict['detection_boxes'][i])
            row.extend( gt_bb )
            rows.append(row)
        return rows
