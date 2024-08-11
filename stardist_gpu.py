import os
import glob

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm
import cv2
import pickle

global BINARY
BINARY = False


class StardistProcessor:

    def __init__(self,  model_type="2D_versatile_he", use_gpu=True):
        self.gpu = use_gpu
        self.model = StarDist2D.from_pretrained(model_type)

        if self.gpu:
            self.model.config.use_gpu = True


    def model_eval(self, image_array, nms_thresh=None, prob_thresh=None):
        """
        Evaluate/Inference the stardist model on image_array.

        the params are based on Stardist model API,
        Using default values: prob_thresh=0.692478, nms_thresh=0.3
        :param image_array:
        :param nms_thresh:
        :param prob_thresh:
        :return:
        """

        # normalize channels jointly
        image_array = normalize(image_array, 1, 99.8, axis=(0, 1, 2))

        # model inference
        labels, res = self.model.predict_instances(
            image_array, nms_thresh=nms_thresh, prob_thresh=prob_thresh
        )

        return labels, res

    def load_image(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        return image


if __name__ == "__main__":

    start_time = time.time()
    # Base directory containing multiple folders of PNG files
    base_image_dir = '/path/to/base/'

    # Subdirectory within the base directory that contains the PNG files
    png_subdir_name = 'folder1'

    # Output directory for predictions
    output_predictions_dir = '/path/to/result'

    # Full path to the folder containing the PNG files
    png_folder_path = os.path.join(base_image_dir, png_subdir_name)


    model = StardistProcessor()

    # inference 
    if not os.path.exists(output_predictions_dir):
        os.makedirs(output_predictions_dir)

    image_files = glob.glob(os.path.join(png_folder_path, '*.png'))

    try:
        for img in image_files:
            image_array = model.load_image(img)

            # Image error
            if np.mean(image_array) < 20:
                print(f"Image error check {img}\n")
                continue

            # Inference 
            labels, res = model.model_eval(image_array)     # instance map, res

            # Save Binary Mask (default: False)
            if BINARY:
                binary_map = ((labels > 0).astype(np.uint8)) * 255
                output_file = os.path.join(output_predictions_dir, os.path.basename(img).replace(".png", "_binary.png"))
                if binary_map.ndim != 3:
                    image = Image.fromarray(np.stack((binary_map, binary_map, binary_map), axis=-1))
                    image.save(output_file)

            # Save Instance Mask as contours.png (contour + image overlay) and .npy File
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]
            for label in unique_labels:
                binary_mask = np.where(labels == label, 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)

            np.save(os.path.join(output_predictions_dir, os.path.basename(img).replace(".png", "_contours.npy")), labels)
            Image.fromarray(image_array).save(os.path.join(output_predictions_dir, os.path.basename(img).replace(".png", "_contours.png")))

    except Exception as e:
        print('Error in  ' + img)


print(time.time() - start_time)


