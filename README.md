# Read Me
A customized GPU inference pipeline for cell nuclei foundation model - StarDist (H&E)

The main script is `stardist_gpu.py`. The detailed description of how to use this GPU-based StarDist codebase is provided below.

## For Reference:
- The StarDist paper: [StarDist Paper](https://arxiv.org/abs/2203.02284)
- The StarDist GitHub codebase: [StarDist GitHub](https://github.com/stardist/stardist)
- A paper that provides an assessment of cell nuclei AI foundation models (including StarDist) on kidney pathology images: [Assessment Paper(TBD)](https://example.com/assessment-paper)

## Requirements
Before running the script, ensure you have the following Python packages installed:
- `numpy`
- `PIL` (Pillow)
- `csbdeep`
- `stardist`
- `tqdm`
- `cv2` (OpenCV)
- `matplotlib`
  
The requirements.txt (or environment.yml) is the exported environment file for my conda environment setup

## Usage

### 1. Configure Input and Output Paths
Modify the script to set your paths:
- **`base_image_dir`**: The base directory containing multiple folders of PNG files.
- **`png_subdir_name`**: The specific folder name within the base directory that contains the PNG files you want to process.
- **`output_predictions_dir`**: The directory where the prediction results will be saved.

### 2. Run the Script

Execute the script by running:

```bash
python stardist_inference.py
```

### 3. Output

For each image in the input folder:
- **Contours Image**: A `.png` file with the original image overlaid with contours of detected instances.
- **Instance Mask**: A `.npy` file containing the instance segmentation map.
- **Binary Mask** (optional): If `BINARY` is set to `True`, a binary mask is also saved as a `.png` file.

### 4. Customizing Inference Parameters

You can customize the inference by modifying the following parameters in the script:
- **`model_type`**: The type of StarDist2D model to use (default is `"2D_versatile_he"`).
- **`use_gpu`**: Set to `True` to enable GPU usage if available.
- **`nms_thresh`**: Non-Maximum Suppression threshold (optional).
- **`prob_thresh`**: Probability threshold for object detection (optional).

### 5. Error Handling

If an image is found to be corrupted or unreadable (e.g., very low average pixel value), the script will skip that image and continue processing the remaining images.

### 6. Performance
The total processing time will be printed to the console after the script has finished running.

