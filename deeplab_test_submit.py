import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from io import BytesIO
import tarfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


input_size = 513
max_epochs = 10
batch_size = 16
orig_width = 1918
orig_height = 1280
threshold = 0.5


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      	if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        	file_handle = tar_file.extractfile(tar_info)
        	graph_def = tf.GraphDef.FromString(file_handle.read())
        	break

    tar_file.close()

    if graph_def is None:
      	raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      	tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = orig_width, orig_height
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    #resized_image = np.resize(image, target_size)
    batch_seg_map = self.sess.run(
         self.OUTPUT_TENSOR_NAME,
         feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)
  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
      ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


LABEL_NAMES = np.asarray(['background', 'car'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


model = DeepLabModel('/models-master/research/deeplab/logdir/train/deeplab_model.tar.gz')
print('model loaded successfully!')



df_test = pd.read_csv('/input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []



print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))


for i in tqdm(range(0,len(ids_test))):
    img = Image.open('/input/test/{}.jpg'.format(ids_test[i]))
    #print ('The type of this data is: ', type(img))
    #img = cv2.resize(img, (input_size, input_size))
    #img = np.array(img, np.float32) / 255
    resized, pred = model.run(img)
    #pred = np.squeeze(pred, axis=1)
    pred = pred.astype('float32')
    prob = cv2.resize(pred, (orig_width, orig_height))
    mask = prob > threshold
    rle = run_length_encode(mask)
    rles.append(rle)


print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
