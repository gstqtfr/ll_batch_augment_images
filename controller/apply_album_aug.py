from controller.album_to_yolo_bb import multi_obj_bb_yolo_conversion
from controller.album_to_yolo_bb import single_obj_bb_yolo_conversion
from controller.save_augs import save_aug_image, save_aug_lab
from controller.validate_results import draw_yolo
import albumentations as A


def apply_aug(image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes, verbose=True):
    transform = A.Compose([
        A.RandomCrop(width=300, height=300),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=-1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),        
        A.Resize(300, 300)
    ], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # JKK: not sure if this is doing the "right thing" here. this is looking at the number of
    # JKK: boundary boxes, rather than the len(transformed_bboxes); what happens if this is
    # JKK: is zero?
    # JKK: hacked the code to make it behave itself
    # TODO: JKK: get rid of the print-as-debug krap below

    transformed_objs = len(transformed)
    print(f"apply_aug: we'll be outputting to file {out_img_pth}")
    print(f"apply_aug: length of transformed objects: {transformed_objs}")

    if transformed_objs != 0:
        if transformed_objs == 1:
            print(f"apply_aug: transformed_bboxes: {transformed_bboxes}")
            print(f"apply_aug: calling single_obj_bb_yolo_conversion")
            transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0]), classes]
            print(f"apply_aug: called single_obj_bb_yolo_conversion")
            print(f"apply_aug: calling save_aug_lab")
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
            print(f"apply_aug: called save_aug_lab")
        if transformed_objs > 1:
            transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
            print(f"apply_aug: calling save_aug_lab")
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
            print(f"apply_aug: called save_aug_lab")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".png")
        draw_yolo(transformed_image, transformed_bboxes)
    else:
        print("label file is empty")
