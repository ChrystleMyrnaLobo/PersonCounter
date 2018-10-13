import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
        #self.loadCategoryIndex()

    def str(self):
        txt = self.model_name
        return txt

    def loadCategoryIndex(self):
        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')
        path_to_labels = os.path.join(self.od_dir, path_to_labels)

        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`. Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        #  str(self.category_index[1]['name']) 
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def run_inference(self, img_dir, out_pkl_file):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.path_to_ckpt = self.model_name + '/frozen_inference_graph.pb'
        self.path_to_ckpt = os.path.join(self.od_dir, self.path_to_ckpt)
        self.loadCategoryIndex()

    def run_inference(self, img_dir, out_pkl_file, filt_pkl_file, class_idx=1):
        """ Run inference and filter out detection for a class (default person)"""
        self.run_inference(img_dir, out_pkl_file)

        with open(out_pkl_file,'rb') as fd1:
            ev_data = pickle.load(fd1)
            # Need to sequentially analyse
            for i in range(1,IMAGE_COUNT): # 655
                # File name without extension
                image_id = str(i).zfill(6)
                dt_dict = ev_data[image_id]
                idx = (dt_dict['detection_classes'] == class_idx)
                dt_dict['num_detections'] = np.count_nonzero(idx)
                dt_dict['detection_boxes'] = dt_dict['detection_boxes'][idx, :]
                dt_dict['detection_scores'] = dt_dict['detection_scores'][idx]
                dt_dict['detection_classes'] = dt_dict['detection_classes'][idx]

            # Store in pickle
            with open(filt_pkl_file,'wb') as fd2:
                pickle.dump(ev_data, fd2)
