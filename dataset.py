import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc
import imageio

def dataset(base_dir, n):
    # print("base_dir : {}, n : {}".format(base_dir, n))
    label_dict = {}
    with open("labels.txt") as f:
        for line in f:
            (key, val) = line.split(',')
            label_dict[key] = val.rstrip()
    #print(label_dict)
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())
    #print(d)
    # print("classes : {}".format(tags))
    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []
    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            file_idx = filename[:-4].split("_")[1]
            processed_image_count += 1
            img = imageio.imread(filename)
            img = img[:,:,:3]
            height, width, chan = img.shape
            assert chan == 3
            X.append(img)
            y.append(float(label_dict[file_idx]))
            useful_image_count += 1
    # print("processed {}, used {}".format(processed_image_count,useful_image_count))

    X = np.array(X).astype(np.float32)

    y = np.array(y)
    print(y)
    return X, y, tags
