from controller.album_to_yolo_bb import multi_obj_bb_yolo_conversion
from controller.album_to_yolo_bb import single_obj_bb_yolo_conversion
from controller.save_augs import save_aug_image, save_aug_lab
from controller.validate_results import draw_yolo
import albumentations as A

def apply_aug(image, shape, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes):

    # JKK: one way of boosting the crap out o this would be to create
    # JKK: a Whole Bunch of transforms & select them at random & then
    # JKK: go ahead & apply them to create a few images

    transform = A.Compose([
        A.RandomCrop(width=300, height=300),
        #A.RandomCrop(width=shape[0], height=shape[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=-1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        #A.Resize(300, 300)
        A.Resize(width=shape[1], height=shape[0])
    ], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # JKK: not sure if this is doing the "right thing" here. this is looking at the number of
    # JKK: boundary boxes, rather than the len(transformed_bboxes); what happens if this is
    # JKK: is zero?
    # JKK: hacked the code to make it behave itself

    transformed_objs = len(transformed)

    if transformed_objs != 0:
        if transformed_objs > 1:
            transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        else:
            transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0]), classes]
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".png")
        draw_yolo(transformed_image, transformed_bboxes)
    else:
        print(f"label file is empty")
