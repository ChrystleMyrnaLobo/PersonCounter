from PIL import Image
import os
import numpy as np
from object_detection.utils import visualization_utils as vis_util

class VisualizeImage:
    """ Draw boundary box on image file """
    def __init__(self, img_dir, out_dir, cat_idx):
        """
        img_dir : Path to image directory (wrt ~/MTP ) E.g: Dataset/Image
        out_dir : Path to save annotated image_size (wrt ~/MTP) E.g: Output/Image
        cat_idx : category index to image class/label
        """
        self.category_index = cat_idx
        self.path_to_image_dir = img_dir
        self.path_to_output_dir = out_dir
        self.str()

    def str(self):
        txt = "Image directory ", self.path_to_image_dir
        txt += " Output directory", self.path_to_output_dir
        return txt

    def draw_BB(self, image_np, bbs, categories, scores, score_thr=0.3, line_thick=3):
        """ Draw detection BB for given image """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            bbs,
            categories,
            scores,
            self.category_index,
            None, #instance_masks=prediction.get('detection_masks'),
            use_normalized_coordinates=False,
            min_score_thresh=score_thr,
            line_thickness=line_thick)

    def load_image_into_numpy_array(self, image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def loadImage(self, image_name):
        """ Return image as numpy array """
        original_image_path = os.path.join(self.path_to_image_dir, image_name)
        image = Image.open(original_image_path)
        image_np = self.load_image_into_numpy_array(image)
        return image_np

    def saveImage(self, image_np, image_name):
        """ Save image image_np as .jpg file """
        annotated_image_path = os.path.join(self.path_to_output_dir, image_name)
        image = Image.fromarray(image_np)
        image.save(annotated_image_path)

    def draw_image(self, image_name, cur_dict, isGT):
        """ Wrapper for input as dictionary by object detection template per image"""
        image_np = self.loadImage(image_name)
        if isGT:
            self.draw_BB(image_np, cur_dict['groundtruth_boxes'], cur_dict['groundtruth_classes'], None)
        else:
            self.draw_BB(image_np, cur_dict['detection_boxes'], cur_dict['detection_classes'], cur_dict['detection_scores'])
        self.saveImage(image_np, image_name)

    def draw_all_images(self, ev_data, isGT):
        #print "Draw BB and save to " + self.path_to_output_dir
        for image_name in os.listdir(self.path_to_image_dir):
            try:
                # image_id : filename without extension # image_name is 000102.jpg
                image_id = os.path.splitext(os.path.basename(image_name))[0]
                self.draw_image(image_name, ev_data[image_id], isGT)
            except Exception as e:
                #print image_id + 'Error' +  e
                continue
