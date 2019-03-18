from utils.dataset import MOT16
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops
from object_detection.utils import per_image_evaluation
import cv2 # 3.4.5

import time, os
import argparse
import pandas as pd
import numpy as np
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger()
logger.setLevel(20) # ignore less than level # Info 20 debug 10

# Override per image evaluation of Tensorflow Object Detection API
class IntersectOverUnion(per_image_evaluation.PerImageEvaluation):
    """Evaluate detection result of a single image.PerImageEvaluation"""
    def __init__(self, num_groundtruth_classes, matching_iou_threshold=0.5, nms_iou_threshold=0.3, nms_max_output_boxes=50, group_of_weight=0.0):
        super(IntersectOverUnion, self).__init__(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold, nms_max_output_boxes)

    def match_frames_on_iou(self, detected_boxlist, groundtruth_boxlist):
        """
            Given two frames, match the detection to groundtruth.
            In decreasing sorted order, assign the GT if IoU > IoU threshold
            Filter detection score < score threshold

            Args:
              detected_boxes: A numpy array of shape [N, 4] representing detected box coordinates
              detected_scores: A 1-d numpy array of length N representing classification score
              groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth box coordinates

            Returns:
              scores: A numpy array representing the detection scores, sorted and filtered.
              max_overlap_gt_ids: A numpy array indicating the detection's corresponding groundtruth box
              tp_fp_labels: a boolean numpy array indicating whether a detection is a true positive.
              is_gt_box_detected: Indicates if a ground truth box is detected
        """
        # detected_boxlist = np_box_list_ops.non_max_suppression(detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
        # Compute IoU for every detection and groundtruth pair. Detection with score < score threshold are ignored. Rest are sorted.
        iou = np_box_list_ops.iou(detected_boxlist, groundtruth_boxlist)
        # The boxlist is sorted, use local_id as unique identifier
        num_detected_boxes = detected_boxlist.num_boxes()

        # Is ith detection a True positive
        tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)
        prev_global_id = groundtruth_boxlist.get_field('global_id')
        cur_global_id = np.full(num_detected_boxes, 0, dtype=float)

        if iou.shape[1] > 0: # For each detection, the best matched GT
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            logger.debug("Matching \n {}".format(max_overlap_gt_ids))
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(0,num_detected_boxes): # i and gt_id are index not local_id
                gt_id = max_overlap_gt_ids[i]
                if iou[i, gt_id] >= self.matching_iou_threshold and not is_gt_box_detected[gt_id]:
                    # assign the person id
                    cur_global_id[i] = prev_global_id[gt_id]
                    tp_fp_labels[i] = True
                    is_gt_box_detected[gt_id] = True
            logger.debug("True positive \n {}".format(tp_fp_labels))
        # Return local_id and global_id mapping
        return (detected_boxlist.get_field('local_id'), cur_global_id)

class AssociateTrack:
    """ Data association between tracks """
    def __init__(self, ds, ipf, opf):
        self.ds = ds
        self.path_to_output_file = ipf
        self.path_to_annotated_file = opf
        self.pie = IntersectOverUnion(1) #TODO

        self.img_cnt = 0
        pass

    def makeBoxList(self, singleFrame):
        """
          Convert single frame into boxlist; to use the Tensorflow Object detection API for IoU matching scripts.
          BoxList represents a list of bounding boxes as numpy array, where each bounding box is represented as a row of 4 numbers, [y_min, x_min, y_max, x_max]. It is assumed that all bounding boxes within a given list correspond to a single image.
          Required field called 'scores'
        """
        bbs = singleFrame[['x', 'y', 'w', 'h']].astype('float')
        bbs['y_max'] = bbs.apply(lambda row: row.y + row.h, axis=1)
        bbs['x_max'] = bbs.apply(lambda row: row.x + row.w, axis=1)
        bbs = bbs[['y', 'x', 'y_max', 'x_max']].values #TODO convert to appropriate type
        local_ids = singleFrame['local_id'].values
        global_ids = singleFrame['global_id'].values
        isNew = singleFrame['isNew'].values

        cnt = singleFrame.shape[0]
        try:
            scores = singleFrame['score'].values
        except KeyError:
            scores = np.full(cnt, 1, dtype=float)
        try:
            classes = singleFrame['class'].values
        except KeyError:
            classes = np.full(cnt, 1, dtype=int)

        frame_boxlist = np_box_list.BoxList(bbs)
        frame_boxlist.add_field('local_id', local_ids)
        frame_boxlist.add_field('global_id', global_ids)
        frame_boxlist.add_field('classes', classes)
        frame_boxlist.add_field('scores', scores)
        frame_boxlist.add_field('isNew', isNew)
        return frame_boxlist

    def updateFrame(self, cur_frame, mapping):
        df2 = pd.DataFrame({'local_id':mapping[0], 'global_id':mapping[1]})
        left_idx = cur_frame.set_index('local_id')
        right_idx = df2.set_index('local_id')
        res = left_idx.reindex(columns=left_idx.columns.union(right_idx.columns))
        res.update(right_idx)
        res.reset_index(inplace=True)
        res['global_id'] = res['global_id'].astype('int64')
        return res

    def matching_algorithm(self, cur_frame, prev_frame, person_count):
        """ Match objects between two frames and correctly tag the person id"""
        #TODO need score
        if cur_frame.shape[0] > 0: # There are BB in prev_frame
            mapping = self.pie.match_frames_on_iou(self.makeBoxList(cur_frame), self.makeBoxList(prev_frame))
            cur_frame = self.updateFrame(cur_frame, mapping)
        else:
            logger.error("No ground truth, All FP")

        # Assign fresh id to unassigned detections
        unassigned_cnt = cur_frame[cur_frame.global_id == 0].shape[0]
        start = person_count + 1
        end = start + unassigned_cnt
        logger.debug("Person count {} Unassigned {} Start {} End {}".format(person_count, unassigned_cnt, start, end))
        fresh_id = [i for i in range(start, end)]
        cur_frame.loc[cur_frame.global_id == 0, 'isNew'] = 1
        cur_frame.loc[cur_frame.global_id == 0, 'global_id'] = fresh_id
        return cur_frame, end

    def assign_id(self):
        """ Traverse all detection per frame and make data association via IoU """
        """ <idx, frame_id, phase, local_id, x, y, w, h, lag>
            16    1         detect  17        809   429  19   61  60
            17    1         detect  18        780   427  21   67  60
            18    61        track   1         1316  476  71  217  11
            19    61        track   2         1477  480  75  179  11
        """
        header = ['frame_id', 'phase', 'local_id', 'x', 'y', 'w', 'h', 'lag']
        dt = pd.read_csv(self.path_to_output_file, index_col = 0, names=header)
        frame_list =  dt.groupby(['frame_id','phase']).size().reset_index().rename(columns={0:'count'})
        dt = dt.assign(global_id=0) # Global person_id
        dt = dt.assign(isNew=0) #is New wrt previous frame
        # Initialize for 1st frame
        pid = dt.loc[dt.frame_id==1,'local_id'].values
        dt.loc[dt.frame_id==1, 'global_id'] = pid
        dt.loc[dt.frame_id==1, 'isNew'] = 1
        person_count = dt.loc[dt.frame_id==1].shape[0]
        prev_fid = 1

        #for fid, phase, count in frame_list[1:-1]: # Upto last but one
        for row in frame_list[1:].itertuples():
            logger.debug("Compare {} frame {} with prev {}".format(row.phase, row.frame_id, prev_fid))
            if row.phase == "detect": # Match with previous tracks
                cur_frame = dt.loc[ dt['frame_id'] == row.frame_id ]
                prev_frame = dt.loc[ dt['frame_id'] == prev_fid ]
                logger.debug("Previous frame \n {}".format(prev_frame))
                logger.debug("Current frame before \n {}".format(cur_frame))
                cur_frame, person_count = self.matching_algorithm(cur_frame, prev_frame, person_count)
                pid = cur_frame['global_id'].values #Convert to numpy
                dt.loc[dt.frame_id==row.frame_id, 'global_id'] = pid
                isNew = cur_frame['isNew'].values #Convert to numpy
                dt.loc[dt.frame_id==row.frame_id, 'isNew'] = isNew
                logger.debug("Current frame updated \n{}".format(dt.loc[ dt['frame_id'] == row.frame_id ]))
            elif row.phase == "track": # Copy the person id wrt previous frame
                dt.loc[dt.frame_id== row.frame_id, 'global_id'] = pid
            else:
                logger.error("Invalid phase")
                exit()
            # for the next iteration
            prev_fid = row.frame_id

        # Save the person_id
        header.append('global_id')
        dt.to_csv(self.path_to_annotated_file, index =False)

    def saveInference(self, frame_id, frame, phase="skip"):
        cv2.imwrite("output/inf/Frame_{}_{}.jpg".format(frame_id, phase), frame)
        # self.img_cnt += 1
        # image_id = str(self.img_cnt).zfill(3)
        # cv2.imwrite("output/inf/Frame_{}.jpg".format(image_id), frame)

    def showInference(self):
        header = ['frame_id', 'phase', 'local_id', 'x', 'y', 'w', 'h', 'lag', 'global_id']
        dt = pd.read_csv(self.path_to_annotated_file)
        prev_fid = 0
        for frame_id in dt.frame_id.unique():
            # Get image
            frame = cv2.imread(self.ds.getFramePath(frame_id))
            # Get detection
            boxes = dt.loc[dt.frame_id == frame_id][['x', 'y', 'w', 'h','global_id', 'isNew']].values
            phase = dt.loc[dt.frame_id == frame_id]['phase'].unique()[0]

            for box in boxes:
                (x, y, w, h) = [int(v) for v in box[0:4]]
                if box[5] == 1: # New object in red
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = str(box[4])
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, label, (x,y), font, 1, (0,225,0), 2, cv2.LINE_AA)

            self.saveInference(frame_id, frame, phase)

            # Show / save only detected and immediate previous frame for IoU
            # if phase == "track":
            #     prev_fid = frame_id
            #     prev_frame = frame
            #     continue
            # self.saveInference(frame_id, frame, phase)
            # if prev_fid != 0:
            #    self.saveInference(prev_fid, prev_frame, "track")

            # cv2.imshow("Frame {} {}".format(phase, frame_id), frame)
            # key = cv2.waitKey(2000)
            # if key == 27: # Esc key
            #     cv2.destroyAllWindows()
            #     break
            # cv2.destroyAllWindows()

if __name__ == '__main__' :
    # python pc_data_association.py -v MOT16-10 -dh ~/4Sem/MTP1/MOT16
    # -i output/localtrack -o output/globalda
    parser = argparse.ArgumentParser()
    parser.add_argument("-dh", "--dataset_home", type=str, help="path to dataset home")
    parser.add_argument("-v", "--video", type=str, default="MOT16-10", help="video stream. e.g: MOT16-10")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="path to dir having local track")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="path to output folder")
    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(str(key) + ': ' + str(value))

    # Dataset
    dataset_name, vid = args.video.split('-')
    if dataset_name == "MOT16":
        ds = MOT16(args.dataset_home, int(vid))

    # filename = "MOT16-10_kcf_pfr0.1_ws5.csv"
    # ipf = "output/localtrack/" + filename
    # opf = "output/globalda/ann_" + filename
    # eve = AssociateTrack(ds,ipf, opf)
    # eve.assign_id()
    # eve.showInference()

    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.csv'):
            continue
        ipf = os.path.join(args.input_dir, filename)
        opf = os.path.join(args.output_dir, filename)
        eve = AssociateTrack(ds, ipf, opf)
        eve.assign_id()

    logger.info("Done")
