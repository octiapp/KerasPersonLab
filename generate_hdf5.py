import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import h5py
from tqdm import tqdm

from config import config

ANNO_FILE = ''
IMG_DIR = ''

coco = COCO(ANNO_FILE)
img_ids = list(coco.imgs.keys())

data = h5py.File(config.H5_DATASET, 'w')
h5_root = data.create_group(name='coco')

for i, img_id in enumerate(tqdm(img_ids)):
    filepath = os.path.join(IMG_DIR, coco.imgs[img_id]['file_name'])
    img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_COLOR)
    h, w, c = img.shape

    crowd_mask = np.zeros((h, w), dtype='bool')
    unannotated_mask = np.zeros((h,w), dtype='bool')
    instance_masks = []
    keypoints = []

    img_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    if len(img_anns) == 0:
        continue
    for anno in img_anns:
        mask = coco.annToMask(anno)

        # if crowd, don't compute loss
        if anno['iscrowd'] == 1:
            crowd_mask = np.logical_or(crowd_mask, mask)
        # if tiny instance, don't compute loss
        elif anno['num_keypoints'] == 0:
            unannotated_mask = np.logical_or(unannotated_mask, mask)
            instance_masks.append(mask)
            keypoints.append(anno['keypoints'])
        else:
            instance_masks.append(mask)
            keypoints.append(anno['keypoints'])

    # Construct encoding:
    
    encoding = np.argmax(np.stack([np.zeros((h,w))]+instance_masks, axis=-1), axis=-1).astype('uint8')
    encoding = np.unpackbits(np.expand_dims(encoding, axis=-1), axis=-1)
    # No image has more than 63 instance annotations, so the first 2 channels are zeros
    encoding[:,:,0] = unannotated_mask.astype('uint8')
    encoding[:,:,1] = crowd_mask.astype('uint8')
    encoding = np.packbits(encoding, axis=-1)

    np_data = np.concatenate([img, encoding], axis=-1)
    this_data = h5_root.create_dataset(name=str(img_id), data=np_data, dtype='uint8')
    this_data.attrs['keypoints'] = keypoints

data.close()