import os
import glob

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from utils import find_files, copy_random_png_images
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
    # path_to_patch_folder: the root directory of all patch (png) folders
    start_time = time.time()
    absolute_input_path = '/home/guoj5/Desktop/wsi-select/Version2_patch_sampled/rodent_kidney_images'
    relative_input_path = 'dataset3_converted'
    absolute_output_path = '/home/guoj5/Desktop/wsi-select/Version2_patch_sampled_predictions/stardist_pred/rodent_kidney_images'
    path_to_patch_folder = os.path.join(absolute_input_path, relative_input_path)

    # list of WSI/patch folder path
    if not os.path.exists(relative_input_path + '_data_dirs.pkl'):
        data_dirs = list(find_files(path_to_patch_folder, format='.png'))
        with open(relative_input_path + '_data_dirs.pkl', 'wb') as binary_file:
            pickle.dump(data_dirs, binary_file)
    else:
        with open(relative_input_path + '_data_dirs.pkl', 'rb') as file:
            data_dirs = pickle.load(file)

    model = StardistProcessor()
    resume_dataset = 0      # resume point

    # inference each folder
    for i in tqdm(range(resume_dataset, len(data_dirs))):

        data_dir = data_dirs[i]
        output_dir = os.path.join(absolute_output_path, data_dir[data_dir.find(relative_input_path):])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = glob.glob(os.path.join(data_dir, '*.png'))

        try:
            for img in image_files:

                image_array = model.load_image(img)

                # Image error
                if np.mean(image_array) < 20:
                    print(f"Image error check {img}\n")
                    continue

                labels, res = model.model_eval(image_array)     # instance map, res

                # Save Binary mask (default: False)
                if BINARY:
                    binary_map = ((labels > 0).astype(np.uint8)) * 255
                    output_file = os.path.join(output_dir, os.path.basename(img).replace(".png", "_binary.png"))
                    if binary_map.ndim != 3:
                        image = Image.fromarray(np.stack((binary_map, binary_map, binary_map), axis=-1))
                        image.save(output_file)

                # Save instance map as contours.png (contour + image overlay) and .npy File (nuclei count 2d matrix)
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels != 0]
                for label in unique_labels:
                    binary_mask = np.where(labels == label, 255, 0).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    cv2.drawContours(image_array, contours, -1, (0, 255, 0), 3)

                np.save(os.path.join(output_dir, os.path.basename(img).replace(".png", "_contours.npy")), labels)
                Image.fromarray(image_array).save(os.path.join(output_dir, os.path.basename(img).replace(".png", "_contours.png")))

        except Exception as e:
            print('Error in  ' + img)


    print(time.time() - start_time)


