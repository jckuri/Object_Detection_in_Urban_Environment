"""
This module is the download process.
"""

import argparse
import io
import os
import subprocess

from PIL import Image
import ray # pylint: disable=import-error
import tensorflow.compat.v1 as tf # pylint: disable=import-error

#from utils import *
import utils


def class_text_to_int(class_text):
    """
    This function maps a class text into an int.
    """
    class_texts = {'vehicle': 1, 'pedestrian': 2, 'cyclist': 4}
    if class_text in class_texts:
        return class_texts[class_text]
    return -1


def create_tf_example(filename, encoded_jpeg, annotations):
    """
    This function create a tf.train.Example from the Waymo frame.

    args:
        - filename [str]: name of the image
        - encoded_jpeg [bytes]: jpeg encoded image
        - annotations [protobuf object]: bboxes and classes

    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """

    # _TODO: Implement function to convert the data

    encoded_jpg_io = io.BytesIO(encoded_jpeg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    filename = filename.encode('utf8')

    for ann in annotations:
        xmin, ymin = ann.box.center_x - 0.5 * \
            ann.box.length, ann.box.center_y - 0.5 * ann.box.width
        xmax, ymax = ann.box.center_x + 0.5 * \
            ann.box.length, ann.box.center_y + 0.5 * ann.box.width
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes.append(ann.type)
        classes_text.append(mapping[ann.type].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/filename': utils.bytes_feature(filename),
        'image/source_id': utils.bytes_feature(filename),
        'image/encoded': utils.bytes_feature(encoded_jpeg),
        'image/format': utils.bytes_feature(image_format),
        'image/object/bbox/xmin': utils.float_list_feature(xmins),
        'image/object/bbox/xmax': utils.float_list_feature(xmaxs),
        'image/object/bbox/ymin': utils.float_list_feature(ymins),
        'image/object/bbox/ymax': utils.float_list_feature(ymaxs),
        'image/object/class/text': utils.bytes_list_feature(classes_text),
        'image/object/class/label': utils.int64_list_feature(classes),
    }))
    return tf_example


def download_tfr(filepath, temp_dir):
    """
    download a single tf record

    args:
        - filepath [str]: path to the tf record file
        - temp_dir [str]: path to the directory where the raw data will be saved

    returns:
        - local_path [str]: path where the file is saved
    """
    # create data dir
    dest = os.path.join(temp_dir, 'raw')
    os.makedirs(dest, exist_ok=True)

    # download the tf record file
    cmd = ['gsutil', 'cp', filepath, f'{dest}']
    utils.logger.info(f'Downloading {filepath}')
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        utils.logger.error(f'Could not download file {filepath}')

    filename = os.path.basename(filepath)
    local_path = os.path.join(dest, filename)
    return local_path


def process_tfr(filepath, data_dir):
    """
    process a Waymo tf record into a tf api tf record

    args:
        - filepath [str]: path to the Waymo tf record file
        - data_dir [str]: path to the destination directory
    """
    # create processed data dir
    dest = os.path.join(data_dir, 'processed')
    os.makedirs(dest, exist_ok=True)
    file_name = os.path.basename(filepath)

    utils.logger.info(f'Processing {filepath}')
    writer = tf.python_io.TFRecordWriter(f'{dest}/{file_name}')
    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
    for idx, data in enumerate(dataset):
        frame = utils.open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        encoded_jpeg, annotations = utils.parse_frame(frame)
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()


@ray.remote
def download_and_process(filename, temp_dir, data_dir):
    """
    This method downloads and process.
    """
    # need to re-import the logger because of multiprocesing
    utils.logger = utils.get_module_logger(__name__)
    local_path = download_tfr(filename, temp_dir)
    process_tfr(local_path, data_dir)
    # remove the original tf record to save space
    utils.logger.info(f'Deleting {local_path}')
    os.remove(local_path)


def main():
    """
    The main method.
    """
    parser = argparse.ArgumentParser(
        description='Download and process tf files')
    parser.add_argument('--data_dir', required=True,
                        help='processed data directory')
    parser.add_argument('--temp_dir', required=True,
                        help='raw data directory')
    args = parser.parse_args()
    logger = utils.get_module_logger(__name__)
    # open the filenames file
    with open('filenames.txt', 'r') as file:
        filenames = file.read().splitlines()
    message = f'Download {len(filenames)} files. Be patient, this will take a long time.'
    logger.info(message)

    data_dir = args.data_dir
    temp_dir = args.temp_dir
    # init ray
    # ray.init(num_cpus=cpu_count())
    ray.init(num_cpus=1)

    workers = [download_and_process.remote(
        fn, temp_dir, data_dir) for fn in filenames[:100]]
    #workers = [download_and_process.remote(fn, temp_dir, data_dir) for fn in filenames[:10]]
    _ = ray.get(workers)


if __name__ == "__main__":
    main()
