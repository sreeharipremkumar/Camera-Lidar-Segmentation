#Sreehari Premkumar
#Northeastern University


import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import cv2
import pathlib

from DeepLabModel import DeepLabModel

def create_colormap():
    
    colormap = np.array([
    [128,  64, 128],
    [244,  35, 232],
    [ 70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [ 70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [  0,   0, 142],
    [  0,   0,  70],
    [  0,  60, 100],
    [  0,  80, 100],
    [  0,   0, 230],
    [119,  11,  32],
    [  0,   0,   0]], dtype=np.uint8)

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


if __name__ == '__main__':
    #Define All Labels for the Model
    LABEL_NAMES = np.asarray([
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle', 'void'])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    #Define the model path directory to be loaded 
    ModelPath = os.path.join( (pathlib.Path(__file__).parent.absolute()) , "Model/deeplab_model.tar.gz")

    #Create instance of DeepLabModel Class and loading the model
    MODEL = DeepLabModel(ModelPath)
    print('model loaded successfully!')


    out_segment = cv2.VideoWriter('Video/output_segmented.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1242,375))
    out_original = cv2.VideoWriter('Video/output_original.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1242,375))

    camera_folder = "dataset/data/image_02/data"
    camera_list = os.listdir(camera_folder)

    # Loop through files in folder
    camera_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for camera_name in camera_list:
        print("Frame: ",camera_name)
        camera_path = os.path.join(camera_folder, camera_name)
        
        img_bgr = cv2.imread(camera_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width,_ = img.shape

        Pil_img = Image.fromarray(img)
        seg_map = MODEL.run(Pil_img)

        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)

        cv2.imshow("Original",img_bgr)
        out_original.write(img_bgr)
        cv2.imshow("Segmented",seg_image)
        out_segment.write(seg_image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            exit()
    out_original.release()
    out_segment.release()
    cv2.destroyAllWindows()
    print('DONE!')

