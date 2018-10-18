from utils.dataset import MOT16
from PIL import Image
import os
import csv

class MOT16_reid:
    """
    Camera setup:
        Two cameras single-shot (one image per camera per person)
        Two cameras multi-shot (More than one image per camera per person)
    """
    def __init__(self, video_id, isSS):
        self.ds = MOT16(video_id)
        self.dataset_name = "MOT16_reid"
        self.total_camera = 2 # For our framework
        self.isSingleShot = isSS
        if self.isSingleShot:
            self.path_to_output_dir = os.path.join(os.pardir, self.dataset_name, 'SSMOT16-' + self.ds.video_seq)
        else: ## TODO: handle for multi shot
            self.path_to_output_dir = os.path.join(os.pardir, self.dataset_name, 'MSMOT16-' + self.ds.video_seq)

        if not os.path.exists(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
            for i in range(1, self.total_camera+1):
                os.makedirs( os.path.join(self.path_to_output_dir, 'cam_'+str(i)) ) # SSMOT16-10/cam_0

    def cropPerson(self, image_id, person_id, cam_id, gt_bb):
        """ Crop the person_id from image_id given by bb as a tuple of x/y coordinates (x1, y1, x2, y2) in cam_id folder"""
        image_name = str(image_id).zfill(6) + '.jpg' # '000001.jpg'
        image_path = os.path.join(self.ds.path_to_image_dir, image_name)
        out_name = os.path.join('cam_' + str(cam_id), str(person_id) + '-' + image_name ) # 1-000001.jpg
        out_path = os.path.join(self.path_to_output_dir, out_name)
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(gt_bb)
        cropped_image.save(out_path)

    def cropAndAdd(self, image_id, person_id, gt_bb, occuranceCnt):
        # Consider the 3st occurance for cam 1 and 10 occurance for camera 2
        if occuranceCnt == 3:
            cam_id = 1
        elif occuranceCnt == 10:
            cam_id = 2
        else:
            return

        # Convert od bb format to format required image api
        [ymin, xmin, ymax, xmax] = gt_bb
        gt_bb = (xmin, ymin, xmax, ymax)
        self.cropPerson(image_id, person_id, cam_id, gt_bb)

    def createGalleryAndProbe(self):
        """ Prepare dataset for the reid framework """
        cam_id = 0 # Image goes to cam_id
        prev_person_id = None # Currently tracked person id to fill
        occuranceCnt = 0 # count the occurance of person

        with open(self.ds.path_to_annotation_dir, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                image_id, cur_person_id, gt_bb, conf, class_idx = self.ds.mot_to_od(row)
                # Ignore the row if confidence flag is 0
                if not conf or image_id > self.ds.image_count:
                    continue

                # On new person, reset # The groundtruth is sorted on the person_id
                if cur_person_id != prev_person_id:
                    prev_person_id = cur_person_id
                    cam_id = 0
                    occuranceCnt = 0

                occuranceCnt += 1
                # Valid frame, so process
                if self.isSingleShot:
                    self.cropAndAdd(image_id, cur_person_id, gt_bb, occuranceCnt)
                ## TODO: multishot
