from PIL import Image
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

    def str(self):
        txt = "Image directory ", self.img_dir
        txt += " Output directory", self.out_dir)
        return txt

    def drawBB(image_np, bbs, cls, line_thick=8):
        """ Draw ground truth BB for given image """
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          bbs,
          cls,
          None,
          self.category_index,
          None, #instance_masks=groundtruth.get('detection_masks'),
          use_normalized_coordinates=False,
          line_thickness=line_thick)

    def drawBB(image_np, bbs, cls, score, score_thr=0.3, line_thick=8):
        """ Draw detection BB for given image """
        image_np,
        bbs,
        cls,
        scores,
        self.category_index,
        None, #instance_masks=prediction.get('detection_masks'),
        use_normalized_coordinates=False,
        min_score_thresh=score_thr,
        line_thickness=line_thick)

    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def loadImage(image_name):
        """ Return image as numpy array """
        original_image_path = os.path.join(self.path_to_image_dir, image_name)
        image = Image.open(original_image_path)
        image_np = load_image_into_numpy_array(image)
        return image_np

    def saveImage(image_np, image_name):
        """ Save image image_np as .jpg file """
        annotated_image_path = os.path.join(self.path_to_output_dir, image_name)
        image = Image.fromarray(image_np)
        image.save(annotated_image_path)

    def draw_gt_image(image_name, gt_dict):
        """ Wrapper for input as groundtruth dictionary by object detection template per image"""
        image_np = self.loadImage(image_name)
        self.drawBB(image_np, gt_dict['groundtruth_boxes'], gt_dict['groundtruth_classes'])
        self.saveImage(image_np)

    def draw_dt_image(image_name, dt_dict):
        """ Wrapper for input as detection dictionary by object detection template per image"""
        image_np = self.loadImage(image_name)
        self.drawBB(image_np, dt_dict['detection_boxes'], dt_dict['detection_classes'], dt_dict['detection_scores'])
        self.saveImage(image_np)
