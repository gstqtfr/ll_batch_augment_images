import os
import pathlib
import cv2
import yaml
from progress.bar import Bar
from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list

with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)

def run_pipeline():
    imgs = os.listdir(CONSTANTS["inp_img_pth"])

    # for i in progressbar(range(100)):

    # JKK: bloody here!

    with Bar('Processing...', max=len(imgs)) as bar:
        for i, img_file in enumerate(imgs):

            # JKK: this won't work with the rather more complicated filenames we have
            # JKK: so let's hack this. pathlib to the rescue, I think
            # file_name = img_file.split('.')[0]

            file_name = pathlib.Path(img_file).stem
            aug_file_name = file_name + "_" + CONSTANTS["transformed_file_name"]

            image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))
            dimensions = image.shape
            lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')

            if not pathlib.Path(lab_pth).is_file():
                print(f"Can't find labels file {lab_pth}, continuing")
            else:
                album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])
                apply_aug(image=image,
                          shape=dimensions,
                          bboxes=album_bboxes,
                          out_lab_pth=CONSTANTS["out_lab_pth"],
                          out_img_pth=CONSTANTS["out_img_pth"],
                          transformed_file_name=aug_file_name,
                          classes=CONSTANTS['CLASSES'])
            bar.next()
