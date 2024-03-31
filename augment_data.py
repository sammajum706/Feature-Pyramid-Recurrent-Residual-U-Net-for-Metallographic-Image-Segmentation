import os
import cv2
from tqdm import tqdm
from albumentations import CenterCrop, GridDistortion, ShiftScaleRotate



def augment_data(images_path, masks_path):

    id = 0
    x_list=os.listdir(images_path)
    y_list=os.listdir(masks_path)
    for x, y in tqdm(zip(x_list, y_list), total=len(x_list)):


        x = cv2.imread(x, 1)
        y = cv2.imread(y, 1)


        x1 = cv2.flip(x, 0 )
        y1 = cv2.flip(y, 0 )


        x2 = cv2.flip(x, 1 )
        y2 = cv2.flip(y, 1 )


        x3 = cv2.flip(x, -1 )
        y3 = cv2.flip(y, -1 )


        x4 = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
        y4 = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)

        x5 = cv2.rotate(x, cv2.ROTATE_180)
        y5 = cv2.rotate(y, cv2.ROTATE_180)


        save_images = [x1, x2, x3, x4, x5]
        save_masks =  [y1, y2, y3, y4, y5]

        idx = 0
        for i, m in zip(save_images, save_masks):


            tmp_img_name =  "image" + str(id) + str(idx) + ".png"
            tmp_mask_name = "mask" +  str(id)+  str(idx) + ".png"

            image_path = os.path.join(images_path, tmp_img_name)
            mask_path = os.path.join(masks_path, tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1
        id += 1

