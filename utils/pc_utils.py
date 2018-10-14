import numpy as np
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops
from object_detection.utils import per_image_evaluation

# Override per image evaluation
class pc_PerImageEvaluation(per_image_evaluation.PerImageEvaluation):
    """Evaluate detection result of a single image.PerImageEvaluation"""
    def __init__(self, num_groundtruth_classes, matching_iou_threshold=0.5, nms_iou_threshold=0.3, nms_max_output_boxes=50,           group_of_weight=0.0):
        super(pc_PerImageEvaluation, self).__init__(num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold, nms_max_output_boxes)

    def _get_overlaps_and_scores_box_mode(
      self,
      detected_boxes,
      detected_scores,
      groundtruth_boxes,
      groundtruth_is_group_of_list):
        """Computes overlaps and scores between detected and groundtruth boxes."""
        detected_boxlist = np_box_list.BoxList(detected_boxes)
        detected_boxlist.add_field('scores', detected_scores)
        detected_boxlist = np_box_list_ops.non_max_suppression(detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)
        # Sorted - need to get the updated version from here
        gt_non_group_of_boxlist = np_box_list.BoxList(groundtruth_boxes[~groundtruth_is_group_of_list])
        #gt_group_of_boxlist = np_box_list.BoxList(groundtruth_boxes[groundtruth_is_group_of_list])
        iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
        #ioa = np.transpose(np_box_list_ops.ioa(gt_group_of_boxlist, detected_boxlist))
        scores = detected_boxlist.get_field('scores')
        num_boxes = detected_boxlist.num_boxes()
        # return iou, ioa, scores, num_boxes, detected_boxlist
        return iou, scores, num_boxes, detected_boxlist

    def boxlist_to_dict(self, detection_boxlist):
        """ Helper to convert boxlist datastructure to dict"""
        dt_dict = {}
        dt_dict['detection_boxes'] = np.array( detection_boxlist.get(), dtype="float32" )
        dt_dict['detection_scores'] = np.array( detection_boxlist.get_field('scores'), dtype="float32" )
        dt_dict['num_detections'] = np.shape(dt_dict['detection_scores'])[0]
        return dt_dict

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
    def match_frames_on_iou(self, dt_dict, groundtruth_boxes, groundtruth_person_id):
        # Return sorted detection dict with corrresponsing ID from gt or NULL
        detected_boxes = dt_dict['detection_boxes']
        detected_scores = dt_dict['detection_scores']

        # Default value false
        num_groundtruth_boxes = np.shape(groundtruth_boxes)[0]
        groundtruth_is_group_of_list = np.zeros(num_groundtruth_boxes, dtype=bool)

        # Compute IoU for every detection and groundtruth pair
        # Detection with score < score threshold are ignored. Rest are sorted.
        (iou, scores, num_detected_boxes, detection_boxlist) = self._get_overlaps_and_scores_box_mode(
                   detected_boxes=detected_boxes,
                   detected_scores=detected_scores,
                   groundtruth_boxes=groundtruth_boxes,
                   groundtruth_is_group_of_list=groundtruth_is_group_of_list)

        # Restore dt
        dt_dict = self.boxlist_to_dict(detection_boxlist)
        dt_dict['person_id'] = np.full( num_detected_boxes, -1)

        # If no GT value then all detection are false positive
        if groundtruth_boxes.size == 0:
            #return scores, np.zeros(num_detected_boxes, dtype=bool)
            print "No ground truth, All FP"
            return dt_dict

        # Is ith detection a True positive
        tp_fp_labels = np.zeros(num_detected_boxes, dtype=bool)

        # Tp-fp evaluation for non-group of boxes (if any).
        if iou.shape[1] > 0:
            # For each detection, the best matched GT
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_box_detected = np.zeros(iou.shape[1], dtype=bool)
            for i in range(num_detected_boxes):
                gt_id = max_overlap_gt_ids[i]
                if iou[i, gt_id] >= self.matching_iou_threshold and not is_gt_box_detected[gt_id]:
                    # assign the person id
                    dt_dict['person_id'][i] = groundtruth_person_id[gt_id] # gt['person_id'][gt_id]
                    tp_fp_labels[i] = True
                    is_gt_box_detected[gt_id] = True

        # Detection matched to
        #for i in range(num_detected_boxes):
            #print "Dt", scores[i], "GT", max_overlap_gt_ids[i], "", tp_fp_labels[i]
        #print "Dt", dt_dict['detection_scores'], "ID", dt_dict['person_id']
        #for i in range(num_groundtruth_boxes):
        #    print "GT", i, bool(is_gt_box_detected[i])
        # convert detected boxlist to dict
        # Assign ID, return dt and tp_fp_labels

        #return scores, max_overlap_gt_ids, tp_fp_labels, is_gt_box_detected
        return dt_dict, np.sum(~is_gt_box_detected)
