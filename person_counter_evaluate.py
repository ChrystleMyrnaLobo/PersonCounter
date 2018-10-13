# # Person Counter: Prediction
# This notebook contains code for prediction using pre-trained models. It stores the output in a pickle file

DATASET = "MOT16"
VIDEO_SEQ_ID = 10 # Range 01 to 14
MODEL_ID = 1
IoU = 0.1 # 0.1, 0.2, 0.3, 0.5, 0.7
useGT = True


# # Imports
import numpy as np
import os
import sys
import tensorflow as tf
import copy

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt # Commented because of warning that matplot lib is already loaded
from PIL import Image
import pickle
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#from pc_utils import pc_PerImageEvaluation
import pc_utils
sys.path.append("../obj_det/")
from utils import label_map_util
from utils import visualization_utils as vis_util

pc_label = {} # For label marking

# Convert from normalized coordinates to original image coordinates
def denormalise_box(box, image_size):
    box[:,0] = box[:,0] * image_size[1]
    box[:,1] = box[:,1] * image_size[0]
    box[:,2] = box[:,2] * image_size[1]
    box[:,3] = box[:,3] * image_size[0]
    return box

# ## Directory Structure
# ```
# MOT16
# /train
#   /MOT16-02
#     /seqinfo.ini
#     /img1
#     /gt
#         gt.txt
#
# PersonCounter
#  /Output
#    /ModelA
#         prediction                 // Pickle file of groundtruth and prediction
#         /Image                     // Folder of images with GT and predicted BB
#         evaluate                   // Results of evalute
#
# ```

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
PERSON_COUNTER = 0

# Unroll the prediction BB as multiple row
#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
def appendMatching(image_id, prediction):
    # Result per frame per object
    for i in range(prediction['num_detections']):
        row = []
        # ID
        row.append(prediction['person_id'][i])

        if useGT:
            print prediction['groundtruth_boxes']
            # bb_left
            row.append(prediction['groundtruth_boxes'][i][1])
            # bb_top
            row.append(prediction['groundtruth_boxes'][i][2])
            # bb_width
            row.append(prediction['groundtruth_boxes'][i][3] - row[2])
            # bb_height
            row.append(row[3] - prediction['groundtruth_boxes'][i][0] )
        else:
            # bb_left
            row.append(prediction['detection_boxes'][i][1])
            # bb_top
            row.append(prediction['detection_boxes'][i][2])
            # bb_width
            row.append(prediction['detection_boxes'][i][3] - row[2])
            # bb_height
            row.append(row[3] - prediction['detection_boxes'][i][0] )
        row = ','.join(map(str,row))

        # Frame number
        row = image_id + ',' + row + '\n'
        with open(RESULT_CSV_FILE,'a') as fd:
            fd.write(row)

    # Summary per frame
    # <frame_id> <total_objects> <entry object> <exited object> <same object>
    # Total object is this frame
    # cnt of objects entered / detected first time wrt prev frame
    # cnt of objects who left the frame wrt prev

    row = []
    # total
    row.append( prediction['num_detections'] )
    # entry_cnt per frame
    row.append( prediction['num_detections_entry'] )
    # exit_cnt per frame
    row.append( prediction['num_detections_exit'] )
    # same
    row.append( row[0] - row[1])

    row = ','.join(map(str, row))
    row = image_id + ',' + row + '\n'
    with open(SUMMARY_CSV_FILE,'a') as fd:
        fd.write(row)

# Match objects between two frames and tag id to new objects
def twoFrameMatching(pie, dt, gt):
    dt, exit_cnt = pc_utils.matchOnIoU(pie, dt, gt, useGT)
    entry_cnt = 0
    # Assign id to unassigned detections
    for i in range(dt['num_detections']):
        if dt['person_id'][i] == -1:
            global PERSON_COUNTER
            PERSON_COUNTER += 1
            global pc_label
            pc_label[PERSON_COUNTER] = { 'id': PERSON_COUNTER, 'name' : 'PC'+ str(PERSON_COUNTER) }
            dt['person_id'][i] = PERSON_COUNTER
            entry_cnt = entry_cnt + 1
    dt['num_detections_entry'] = entry_cnt
    dt['num_detections_exit'] = exit_cnt

    return dt

def makeGT(image_id, gt, initMode=False):
    if initMode:
        # Initialize the person counter
        global PERSON_COUNTER
        PERSON_COUNTER = gt['num_detections']

        gt['person_id'] = np.array([i+1 for i in range(gt['num_detections'])])
        # fill the pc label map
        for i in range(gt['num_detections']):
            global pc_label
            pc_label[i] = { 'id': i, 'name' : 'PC'+ str(i)}

    appendMatching(image_id, gt)
    # Save first image
    #image_path = image_id + '.jpg'
    #drawBB(image_path, gt, isIDMode=True)

    return gt

# Assign ID to person detected
def assignID():
    if os.path.exists(RESULT_CSV_FILE):
        os.remove(RESULT_CSV_FILE)
    if os.path.exists(SUMMARY_CSV_FILE):
        os.remove(SUMMARY_CSV_FILE)

    if useGT:
        # Load gt as
        ev_data = extractGT()
    else:
        with open(FILTERED_PKL_FILE,'rb') as fd:
            ev_data = pickle.load(fd)

    # Init per image evaluation
    num_groundtruth_classes = 1
    matching_iou_threshold = IoU
    nms_iou_threshold = 1.0
    nms_max_output_boxes = 10000

    pie = pc_utils.pc_PerImageEvaluation(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,nms_max_output_boxes)

    # First frame
    frame_id = 1
    image_id = str(frame_id).zfill(6)

    gt = ev_data[image_id] # treated as gt
    gt['num_detections_entry'] = gt['num_detections']
    gt['num_detections_exit'] = 0
    gt = makeGT(image_id, gt, initMode=True)

    # Next frame
    image_id = str(frame_id+1).zfill(6)
    dt = ev_data[image_id] # treated as dt

    for frame_id in range(1,IMAGE_COUNT-1): # Upto last but one
        # Returns gt for next
        gt = twoFrameMatching(pie, dt, gt)
        gt = makeGT(image_id, gt, initMode=False)

        # Prepare next loop
        frame_id = frame_id + 1
        image_id = str(frame_id+1).zfill(6)
        dt = ev_data[image_id]



def main():
    #filter_prediction()
    assignID()
    print "Total person", PERSON_COUNTER
    print "Done"
main()
