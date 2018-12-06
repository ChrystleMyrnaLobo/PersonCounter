import os
import numpy as np
import sys
sys.path.append('../models/research/')
from object_detection.utils import label_map_util
# common utitlity of object detection api

def load_category_index(file_label_map, num_classes):
    """ Dataset is annotated on these category """
    od_dir = os.path.join(os.pardir, 'obj_det')
    # List of the strings that is used to add correct label for each box.
    path_to_labels = os.path.join('data', file_label_map)
    path_to_labels = os.path.join(od_dir, path_to_labels)

    # Label maps map indices to category names, so that when our convolution network predicts `5`,
    # we know that this corresponds to `airplane`. Here we use internal utility functions,
    # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    #  str(category_index[1]['name'])
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def denormalise_box(box, image_size):
    """ Convert from normalized coordinates to original image coordinates """
    box[:,0] = box[:,0] * image_size[1]
    box[:,1] = box[:,1] * image_size[0]
    box[:,2] = box[:,2] * image_size[1]
    box[:,3] = box[:,3] * image_size[0]
    return box

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
