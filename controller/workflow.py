from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list
import cv2
import os
import yaml
import pathlib


with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)


def run_pipeline():
    imgs = os.listdir(CONSTANTS["inp_img_pth"])   
    for img_file in imgs:
        # JKK: this won't work with the rather more complicated filenames we have
        # JKK: so let's hack this. pathlib to the rescue, I think
        #file_name = img_file.split('.')[0]
        file_name = pathlib.Path(img_file).stem
        aug_file_name = file_name + "_" + CONSTANTS["transformed_file_name"]
        image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))           
        lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')
        if not pathlib.Path(lab_pth).is_file():
            print(f"Can't find labels file {lab_pth}, continuing")
        else:
            print(f"Working on image file {img_file}, & label file {lab_pth}")
            album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
            apply_aug(image, album_bboxes, CONSTANTS["out_lab_pth"],  CONSTANTS["out_img_pth"], aug_file_name, CONSTANTS['CLASSES'])
