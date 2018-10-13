import os
from utils.dataset import MOT16
from utils.model import ODModel
#from utils import visualize

class BasicPersonCounter:
    def __init__(self, useGT, v_id, m_idx, iou):
        self.ds = MOT16(v_id)
        print self.ds.str()
        self.model = ODModel(1)
        print self.model.str()
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

        self.path_to_result_csv = os.path.join(self.path_to_output_dir, "result_person.csv") # Per frame per object id
        self.path_to_summary_csv = os.path.join(self.path_to_output_dir, "result_frame.csv") # Frame wise summary

        # Store results to Output/Model directory
        if not os.path.exists(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
            os.makedirs(os.path.join(self.path_to_output_dir,"Image"))

    def str(self):
        txt = "Basic person counter - str\n"
        txt += self.path_to_output_dir + "\n"
        txt += self.path_to_filtered_pkl + "\n"
        txt += self.path_to_summary_csv + "\n"
        return txt

    def run_detection(self):
        """ Run inference for dataset using model and store in pikle file"""
        # filter out detection of other class
        # Person class = 1 in COCO dataset
        self.model.run_inference(self.dataset.path_to_image_dir, self.path_to_prediction_pkl, self.path_to_filtered_pkl, 1)

    def reset_env(self):
        """Remove previous log and load env data"""
        if os.path.exists(self.path_to_result_csv):
            os.remove(self.path_to_result_csv)
        if os.path.exists(self.path_to_summary_csv):
            os.remove(self.path_to_summary_csv)
        self.person_counter = 0
        self.pc_label = {}
        if self.useGT:
            ev_data = self.ds.parseGroundtruth()
        else:
            with open(self.path_to_filtered_pkl,'rb') as fd:
                ev_data = pickle.load(fd)

        return ev_data

    # Unroll the prediction BB as multiple row
    #<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    def appendMatching(self, image_id, dt_dict):
        # Result per frame per object
        for i in range(dt_dict['num_detections']):
            row = []
            # ID
            row.append(dt_dict['person_id'][i])

            if useGT:
                print dt_dict['groundtruth_boxes']
                # bb_left
                row.append(dt_dict['groundtruth_boxes'][i][1])
                # bb_top
                row.append(dt_dict['groundtruth_boxes'][i][2])
                # bb_width
                row.append(dt_dict['groundtruth_boxes'][i][3] - row[2])
                # bb_height
                row.append(row[3] - dt_dict['groundtruth_boxes'][i][0] )
            else:
                # bb_left
                row.append(dt_dict['detection_boxes'][i][1])
                # bb_top
                row.append(dt_dict['detection_boxes'][i][2])
                # bb_width
                row.append(dt_dict['detection_boxes'][i][3] - row[2])
                # bb_height
                row.append(row[3] - dt_dict['detection_boxes'][i][0] )
            row = ','.join(map(str,row))

            # Frame number
            row = image_id + ',' + row + '\n'
            with open(self.path_to_result_csv,'a') as fd:
                fd.write(row)

        # Summary per frame
        # <frame_id> <total_objects> <entry object> <exited object> <same object>
        # Total object is this frame
        # cnt of objects entered / detected first time wrt prev frame
        # cnt of objects who left the frame wrt prev

        row = []
        # total
        row.append( dt_dict['num_detections'] )
        # entry_cnt per frame
        row.append( dt_dict['num_detections_entry'] )
        # exit_cnt per frame
        row.append( dt_dict['num_detections_exit'] )
        # same
        row.append( row[0] - row[1])

        row = ','.join(map(str, row))
        row = image_id + ',' + row + '\n'
        with open(self.path_to_summary_csv,'a') as fd:
            fd.write(row)

    def makeGT(self, image_id, gt, initMode=False):
        if initMode:
            # Initialize the person counter
            self.person_counter = gt['num_detections']

            gt_dict['person_id'] = np.array([i+1 for i in range(gt_dict['num_detections'])])
            # fill the pc label map
            for i in range(gt_dict['num_detections']):
                pc_label[i] = { 'id': i, 'name' : 'PC'+ str(i)}

        appendMatching(image_id, gt)
        return gt_dict

    # Match objects between two frames and tag id to new objects
    def twoFrameMatching(self, pie, dt_dict, gt_dict):
        dt_dict, exit_cnt = pc_utils.matchOnIoU(pie, dt_dict, gt_dict, self.useGT)
        entry_cnt = 0
        # Assign id to unassigned detections
        for i in range(dt_dict['num_detections']):
            if dt['person_id'][i] == -1:
                self.person_counter += 1
                self.pc_label[self.person_counter] = { 'id': self.person_counter, 'name' : 'PC'+ str(self.person_counter) }
                dt_dict['person_id'][i] = self.person_counter
                entry_cnt = entry_cnt + 1
        dt_dict['num_detections_entry'] = entry_cnt
        dt_dict['num_detections_exit'] = exit_cnt
        return dt_dict

    def assign_id(self):
        ev_data = reset_env()

        # Init per image evaluation
        num_groundtruth_classes = 1
        matching_iou_threshold = self.iou_thr
        nms_iou_threshold = 1.0
        nms_max_output_boxes = 10000

        pie = pc_utils.pc_PerImageEvaluation(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,nms_max_output_boxes)

        # First frame
        frame_id = 1
        image_id = str(frame_id).zfill(6)

        gt_dict = ev_data[image_id] # treated as gt
        gt_dict['num_detections_entry'] = gt_dict['num_detections']
        gt_dict['num_detections_exit'] = 0
        gt_dict = makeGT(image_id, gt_dict, initMode=True)

        # Next frame
        image_id = str(frame_id+1).zfill(6)
        dt_dict = ev_data[image_id] # treated as dt

        for frame_id in range(1,IMAGE_COUNT-1): # Upto last but one
            # Returns gt for next
            gt_dict = twoFrameMatching(pie, dt_dict, gt_dict)
            gt_dict = makeGT(image_id, gt_dict, initMode=False)

            # Prepare next loop
            frame_id = frame_id + 1
            image_id = str(frame_id+1).zfill(6)
            dt_dict = ev_data[image_id]
