import os
import csv
from utils.dataset import MOT16
from utils.model import ODModel
import numpy as np
from utils.pc_utils import pc_PerImageEvaluation
from utils.visualize import VisualizeImage

class BasicPersonCounter:
    def __init__(self, useGT, v_id, m_idx, iou):
        self.ds = MOT16(v_id)
        self.model = ODModel(1)
        self.iou_thr = iou
        self.useGT = useGT
        self.person_counter = 0
        self.pc_label = {}

        # Output/Model_MOT16_01
        self.path_to_output_dir = os.path.join('Output', self.model.model_name + '_' + self.ds.dataset_name + '-' + self.ds.video_seq)
        self.path_to_prediction_pkl = os.path.join(self.path_to_output_dir, "prediction")
        self.path_to_filtered_pkl = os.path.join(self.path_to_output_dir, "prediction_filtered") # Filtered to only person class

        if useGT:
            self.path_to_output_dir = os.path.join(self.path_to_output_dir, 'gt_Iou' + str(iou))
        else:
            self.path_to_output_dir = os.path.join(self.path_to_output_dir, 'dt_Iou' + str(iou))

        self.path_to_resobj_csv = os.path.join(self.path_to_output_dir, "result_person.csv") # Per frame per object id
        self.path_to_resframe_csv = os.path.join(self.path_to_output_dir, "result_frame.csv") # Frame wise summary

        # Store results to Output/Model directory
        if not os.path.exists(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
            os.makedirs(os.path.join(self.path_to_output_dir,"Image"))

    def str(self):
        txt = "Basic person counter - str\n"
        txt += self.path_to_output_dir + "\n"
        txt += self.path_to_filtered_pkl + "\n"
        txt += self.path_to_resframe_csv + "\n"
        return txt

    def visualize_groundtruth(self):
        """ Draw groundtruth BB on all image """
        # Load data as GT and draw BB - class labels won't be seen
        # ev_data = self.ds.parseGroundtruth(asDetection=False) # Use dataset category_index
        # viz = VisualizeImage(self.ds.path_to_image_dir, os.path.join(self.path_to_output_dir,"Image"), self.ds.category_index)
        # viz.draw_all_images(ev_data, isGT=True)

        # Load data as dt with MOT16 classes
        ev_data = self.ds.parseGroundtruth(asDetection=True) # Use model category_index
        viz = VisualizeImage(self.ds.path_to_image_dir, os.path.join(self.path_to_output_dir,"Image"), self.ds.category_index)
        viz.draw_all_images(ev_data, isGT=False)

    def visualize_reid(self, ev_data):
        # Data as dt with person id as classes
        viz = VisualizeImage(self.ds.path_to_image_dir, os.path.join(self.path_to_output_dir,"Image"), self.pc_label)
        viz.draw_all_images(ev_data, isGT=False)

    def run_inference(self):
        """ Run inference for dataset using model and store in pikle file"""
        # filter out detection of other class
        # Person class = 1 in COCO dataset
        self.model.run_inference(self.dataset.path_to_image_dir, self.path_to_prediction_pkl, self.path_to_filtered_pkl, 1)

    def initialize_base_frame(self, image_id, base_dict):
        """All objects in first frame are unique counts"""
        base_dict['num_detections_entry'] = base_dict['num_detections']
        base_dict['num_detections_exit'] = 0
        self.person_counter = base_dict['num_detections']
        base_dict['detection_classes'] = np.array([i for i in range(1, base_dict['num_detections'] + 1)])
        for i in range(1, base_dict['num_detections']+1):
            self.pc_label[i] = { 'id': i, 'name' : 'PC'+ str(i)}
        self.save_frame(image_id, base_dict)
        return base_dict

    def reset_env(self):
        """Clear log, load data, initialze base frame"""
        if os.path.exists(self.path_to_resobj_csv):
            os.remove(self.path_to_resobj_csv)
        if os.path.exists(self.path_to_resframe_csv):
            os.remove(self.path_to_resframe_csv)
        self.person_counter = 0
        self.pc_label = {}
        # All data association works with detection dictionary format
        if self.useGT:
            ev_data = self.ds.parseGroundtruth(asDetection=True)
        else:
            with open(self.path_to_filtered_pkl,'rb') as fd:
                ev_data = pickle.load(fd)

        # Init per image evaluation
        num_groundtruth_classes = 1
        matching_iou_threshold = self.iou_thr
        nms_iou_threshold = 1.0
        nms_max_output_boxes = 10000
        pie = pc_PerImageEvaluation(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,nms_max_output_boxes)

        # Initialize for first frame
        frame_id = 1
        image_id = str(frame_id).zfill(6)
        base_dict = ev_data[image_id] # treated as gt
        base_dict = self.initialize_base_frame(image_id, base_dict)
        ev_data[image_id] = base_dict
        return ev_data, pie, base_dict

    def save_frame(self, image_id, cur_dict):
        """Save frame content as per object and per frame"""
        # Per object log
        rows = self.ds.parseDetection(image_id, cur_dict)
        with open(self.path_to_resobj_csv,'a') as fd:
            writer = csv.writer(fd, delimiter=",")
            writer.writerows(rows)

        # Per frame log
        # <frame_id> <total_objects> <entry object> <exited object> <same object>
        # Total objects in this frame
        # cnt of objects entered / detected first time wrt prev frame
        # cnt of objects who left the frame wrt prev
        row = []
        row.append( cur_dict['num_detections'] ) # total
        row.append( cur_dict['num_detections_entry'] ) # entry_cnt per frame
        row.append( cur_dict['num_detections_exit'] ) # exit_cnt per frame
        row.append( row[0] - row[1]) # cnt of people who are still there wrt prev frame
        row = ','.join(map(str, row))
        row = image_id + ',' + row + '\n'
        with open(self.path_to_resframe_csv,'a') as fd:
            fd.write(row)

    def two_frame_reid(self, pie, cur_dict, prev_dict):
        """ Match objects between two frames and tag matched id to new objects"""
        cur_dict, exit_cnt = pie.match_frames_on_iou(cur_dict, prev_dict['detection_boxes'], prev_dict['detection_classes'])
        entry_cnt = 0
        # Assign fresh id to unassigned detections
        for i in range(cur_dict['num_detections']):
            if cur_dict['detection_classes'][i] == -1:
                self.person_counter += 1
                self.pc_label[self.person_counter] = { 'id': self.person_counter, 'name' : 'PC'+ str(self.person_counter) }
                cur_dict['detection_classes'][i] = self.person_counter
                entry_cnt = entry_cnt + 1
        cur_dict['num_detections_entry'] = entry_cnt
        cur_dict['num_detections_exit'] = exit_cnt
        return cur_dict

    def assign_id(self):
        """ Traverse all detection per frame and make data association via IoU """
        ev_data, pie, prev_dict = self.reset_env()
        for frame_id in range(2,self.ds.image_count+1): # Upto last but one
            image_id = str(frame_id).zfill(6)
            cur_dict = ev_data[image_id] # Current frame is detection, previous frame is groundtruth
            cur_dict = self.two_frame_reid(pie, cur_dict, prev_dict)
            #ev_data[image_id] = cur_dict
            self.save_frame(image_id, cur_dict)
            # for the next iteration
            prev_dict = cur_dict
        #self.visualize_reid(ev_data)
