import os
import pickle
from utils.od_utils import *

class ODModel:
    """ Object detection DL model used for detection """
    def __init__(self, model_idx):
        # Models used in anupam's paper
        ALL_MODEL = ['ssd_mobilenet_v1_coco_2017_11_17' #0
            ,'ssd_inception_v2_coco_2017_11_17' #1
            ,'rfcn_resnet101_coco_2018_01_28' #2
            ,'faster_rcnn_resnet101_coco_2018_01_28' #3
            ,'faster_rcnn_inception_v2_coco_2018_01_28' #4
        ]
        self.model_name = ALL_MODEL[model_idx]
        self.od_dir = os.path.join(os.pardir, 'obj_det')
        self.num_classes = 90
        self.category_index = load_category_index('mscoco_label_map.pbtxt', self.num_classes) # Model is trained on these categories

    def str(self):
        txt = self.model_name
        return txt

    def run_inference(self, img_dir, out_pkl_file):
        """ Run inference for image and save result in pickle file"""
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.path_to_ckpt = self.model_name + '/frozen_inference_graph.pb'
        self.path_to_ckpt = os.path.join(self.od_dir, self.path_to_ckpt)
        eng = tf_inference.InferenceEngine(self.path_to_ckpt)
        ev_data = {}
        for image_path in os.listdir(img_dir):
            try:
                # image_id : filename without extension
                image_id = os.path.splitext(os.path.basename(image_path))[0]
                image_path = os.path.join(img_dir, image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = eng.run_inference_for_single_image(image_np)
                # The predicition gives BB in normalized coordinated
                # Convert to original image cordinates from normalized coordinates (for evaluation and vizualization)
                output_dict['detection_boxes'] = denormalise_box(output_dict['detection_boxes'], image.size)
                ev_data[image_id] = output_dict
                print image_id, "Detected", output_dict['num_detections']
            except Exception as e:
                print image_id, 'Error', e
                continue
        # Save detections
        with open(out_pkl_file,'wb') as fd:
            pickle.dump(ev_data, fd)
        return ev_data

    def run_inference(self, img_dir, out_pkl_file, filt_pkl_file, class_idx=1):
        """ Run inference and filter out detection for a class (default person)"""
        ev_data = self.run_inference(img_dir, out_pkl_file)

        # with open(out_pkl_file,'rb') as fd:
        #     ev_data = pickle.load(fd)

        for image_id in ev_data:
            # File name without extension
            dt_dict = ev_data[image_id]
            idx = (dt_dict['detection_classes'] == class_idx)
            dt_dict['num_detections'] = np.count_nonzero(idx)
            if dt_dict['num_detections'] == 0:
                del ev_data[image_id]
            else :
                dt_dict['detection_boxes'] = dt_dict['detection_boxes'][idx, :]
                dt_dict['detection_scores'] = dt_dict['detection_scores'][idx]
                dt_dict['detection_classes'] = dt_dict['detection_classes'][idx]
                ev_data[image_id] = dt_dict
        # Store in pickle
        with open(filt_pkl_file,'wb') as fd:
            pickle.dump(ev_data, fd)
        return ev_data
