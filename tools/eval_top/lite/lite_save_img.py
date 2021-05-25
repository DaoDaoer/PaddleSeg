# Python保存数据示例
import numpy as np
import os
from PIL import Image

import paddleseg.transforms as T


def process_image(img):
    transforms = [T.Resize((192, 192)), T.Normalize()]
    t = T.Compose(transforms)
    return t(img)


val_path = 'data/mini_supervisely/val.txt'
bin_file_path = './deploy/val_imgs'
data_dir = 'data/mini_supervisely'

with open(val_path, 'r') as f:
    lines = f.readlines()
    num_images = len(lines)
    with open(bin_file_path, "w+b") as of:
        of.seek(0)
        num = np.array(int(num_images)).astype('int64')
        of.write(num.tobytes())

        for idx, line in enumerate(lines):
            print(idx)
            img_path, label = line.split()

            #save image(float32) to file
            img_path = os.path.join(data_dir, img_path)
            # img = Image.open(img_path)
            data = process_image(img_path)
            img = data[0]
            np_img = np.array(img)
            #of.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 * idx)
            of.write(np_img.astype('float32').tobytes())
