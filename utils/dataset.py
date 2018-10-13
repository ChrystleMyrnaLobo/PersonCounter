import os
import csv
import numpy as np

class MOT16:
    """
        Some observation of MOT16 dataset:
        CSV text-file containing one object instance per line. Each line must contain 10 values:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        All frame numbers, target IDs and bounding boxes are 1-basedself.
        Frame number is image number without leading 0. Person ID is <id>
    """
    def __init__(self, video_id):
        self.dataset_name = "MOT16"
        self.video_seq = str(video_id).zfill(2)
        if video_id in [2,4,5,9,10,11,13]:
            #self.path_to_dataset_dir = '../MOT16/train/MOT16-' + self.video_seq + '/'
            self.path_to_dataset_dir = os.path.join( os.pardir, self.dataset_name, 'train', 'MOT16-' + self.video_seq)
        else:
            #self.path_to_dataset_dir = '../MOT16/test/MOT16-' + self.video_seq + '/'
            self.path_to_dataset_dir = os.path.join( os.pardir, self.dataset_name, 'test', 'MOT16-' + self.video_seq)
        self.image_count = self.getImageCount()
        self.path_to_image_dir = os.path.join(self.path_to_dataset_dir, 'img1')
        self.path_to_anotation_dir = os.path.join(self.path_to_dataset_dir, 'gt', 'gt.txt')

    def str(self):
        txt = "class MOT16\n"
        txt += self.path_to_dataset_dir + "\n"
        txt += self.path_to_anotation_dir + "\n"
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
        return 5
        #return int( meta[idx+10:idx+13] )

    def parseGroundtruth(self, asDetection=False): #extractGT
        """" Read the groundtruth into a dictionary as per OD API format. Return as dt_dict if asDetection is true """
        gt_ds = {} # groundtruth for all images in dataset
        cat_label = "person" # MOT16 has only person class

        with open(self.path_to_anotation_dir, 'rb') as csvfile:
            print "Reading from" + self.path_to_anotation_dir
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                gt_dict = {}
                # For each BB in each image
                gt_bb = []
                gt_bb.append( float(row[3]) ) # ymin # bb_top
                gt_bb.append( float(row[2]) ) # xmin # bb_left
                gt_bb.append( float(row[3]) + float(row[5]) ) # ymax # bb_top + bb_height
                gt_bb.append( float(row[2]) + float(row[4]) ) # xmax # bb_left + bb_width
                image_id = str(row[0]).zfill(6)

                if image_id in gt_ds :
                    gt_dict = gt_ds[image_id]
                    gt_dict['num_detections'] += 1
                    np.append( gt_dict['person_id'], [ row[1] ] )
                    if asDetection:
                        gt_dict['detection_boxes'] = np.append( gt_dict['detection_boxes'], [gt_bb], axis=0)
                        gt_dict['detection_classes'] = np.append( gt_dict['detection_classes'], [cat_label])
                        gt_dict['detection_scores'] = np.append( gt_dict['detection_scores'], [1])
                    else :
                        gt_dict['groundtruth_boxes'] = np.append( gt_dict['groundtruth_boxes'], gt_bb )
                        gt_dict['groundtruth_classes'] = np.append( gt_dict['groundtruth_classes'], [cat_label] )
                        gt_dict['goundtruth_scores'] = np.append( gt_dict['goundtruth_scores'], [1])
                else:
                    gt_dict['num_detections'] = 1
                    gt_dict['person_id'] = np.array( [ row[1] ] )
                    if asDetection:
                        gt_dict['detection_boxes'] = np.array( [gt_bb], dtype="float32")
                        gt_dict['detection_classes'] = np.array( [cat_label] )
                        gt_dict['detection_scores'] = np.array([1])
                    else:
                        gt_dict['groundtruth_boxes'] = np.array( [gt_bb] , dtype="float32")
                        gt_dict['groundtruth_classes'] = np.array( [cat_label] )
                        gt_dict['goundtruth_scores'] = np.array([1])
                gt_ds[image_id] = gt_dict
        return gt_ds

    def parseDetection(self, image_id, dt_dict):
        """ Convert detection dict into csv rows as per MOT16 format"""
        rows = []
        for i in range(0,dt_dict['num_detections']):
            row = []
            row.append(dt_dict['person_id'][i]) # ID
            row.append(dt_dict['detection_boxes'][i][1]) # bb_left #xmin
            row.append(dt_dict['detection_boxes'][i][0]) # bb_top # ymin
            row.append(dt_dict['detection_boxes'][i][3] - row[1]) # bb_width # x_max - xmin
            row.append(dt_dict['detection_boxes'][i][2] - row[2]) # bb_height # ymax - ymin
            row = ','.join(map(str,row))
            row = image_id + ',' + row + '\n'
            rows.append(row)
        return rows
