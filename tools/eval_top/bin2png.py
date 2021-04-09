import numpy as np
import os
from PIL import Image
import struct
from paddleseg import utils

val_path = 'data/mini_supervisely/val.txt'
bin_file_path = './deploy/not_fix_shape_argmax/predict.bin'

save_dir = './deploy/not_fix_shape_argmax/predict'

nx, nz = 192, 192
channel = 2


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


# with open(val_path, 'r') as f:
#     lines = f.readlines()
#     num_images = len(lines)

#     with open(bin_file_path, "rb") as of:
#         pic = np.zeros((channel, nx, nz), dtype=np.float32)
#         count = 0
#         for n in range(num_images):
#             for k in range(channel):
#                 for i in range(nx):
#                     for j in range(nz):
#                         data = of.read(4)
#                         elem = struct.unpack("f", data)[0]
#                         pic[k][i][j] = elem

#             pred = np.argmax(pic, axis=0)
#             line = lines[n]
#             img_path, label = line.split()
#             pred_mask = utils.visualize.get_pseudo_color_map(pred)
#             pred_saved_path = os.path.join(save_dir, img_path.rsplit(".")[0] + ".png")
#             mkdir(pred_saved_path)
#             pred_mask.save(pred_saved_path)
#             count += 1
#             print(count)

# has argmax before
with open(val_path, 'r') as f:
    lines = f.readlines()
    num_images = len(lines)

    with open(bin_file_path, "rb") as of:
        pic = np.zeros((nx, nz), dtype=np.int64)
        count = 0
        for n in range(num_images):
            for i in range(nx):
                for j in range(nz):
                    data = of.read(8)
                    elem = struct.unpack("q", data)[0]
                    pic[i][j] = elem

            line = lines[n]
            img_path, label = line.split()
            pred_mask = utils.visualize.get_pseudo_color_map(pic)
            pred_saved_path = os.path.join(save_dir,
                                           img_path.rsplit(".")[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)
            count += 1
            print(count)
