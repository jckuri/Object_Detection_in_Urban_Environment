"""
This module render the inference video.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt # pylint: disable=import-error
import tensorflow as tf # pylint: disable=import-error
from matplotlib import animation # pylint: disable=import-error

from object_detection.builders.dataset_builder import build as build_dataset # pylint: disable=import-error
from object_detection.utils.config_util import get_configs_from_pipeline_file # pylint: disable=import-error
from object_detection.utils.label_map_util import create_category_index_from_labelmap # pylint: disable=import-error
from object_detection.utils import visualization_utils as viz_utils # pylint: disable=import-error

import utils


def main(labelmap_path, model_path, tf_record_path, config_path, output_path):
    """
    Use a model and a tf record file and create a mp4 video
    args:
    - labelmap_path [str]: path to labelmap file
    - model_path [str]: path to exported model
    - tf_record_path [str]: path to tf record file to visualize
    - config_path [str]: path to config file
    - output_path [str]: path to mp4 file

    Save the results as mp4 file
    """
    # load label map
    category_index = create_category_index_from_labelmap(labelmap_path,
                                                         use_display_name=True)

    # Load saved model and build the detection function
    message = f'Loading model from {model_path}'
    LOGGER.info(message)
    detect_fn = tf.saved_model.load(model_path)

    # open config file
    message = f'Loading config from {config_path}'
    LOGGER.info(message)
    configs = get_configs_from_pipeline_file(config_path)
    #eval_config = configs['eval_config']
    eval_input_config = configs['eval_input_config']
    #model_config = configs['model']

    # update the eval config file
    eval_input_config.tf_record_input_reader.input_path[:] = [tf_record_path]
    dataset = build_dataset(eval_input_config)

    # build dataset
    dataset = build_dataset(eval_input_config)

    # here we infer on the entire dataset
    images = []
    message = f'Inference on {tf_record_path}'
    LOGGER.info(message)
    for idx, batch in enumerate(dataset):
        if idx % 50:
            message = f'Step: {idx}'
            LOGGER.info(message)
        # add new axis and feed into model
        input_tensor = batch['image']
        image_np = input_tensor.numpy().astype(np.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        # tensor -> numpy arr, remove one dimensions
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, ...].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        images.append(image_np_with_detections)

    # now we can create the animation
    figure = plt.figure()
    figure.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None)
    axis = plt.subplot(111)
    axis.axis('off')
    im_obj = axis.imshow(images[0])

    def animate(idx):
        image = images[idx]
        im_obj.set_data(image)

    anim = animation.FuncAnimation(figure, animate, frames=198)
    anim.save(output_path, fps=5, dpi=300)


def main_method():
    """
    The main method.
    """
    parser = argparse.ArgumentParser(description='Create video')
    parser.add_argument('--labelmap_path', required=True, type=str,
                        help='path to the label map')
    parser.add_argument('--model_path', required=True, type=str,
                        help='path to the saved model folder')
    parser.add_argument('--tf_record_path', required=True, type=str,
                        help='path to the tf record file')
    parser.add_argument('--config_path', required=False, type=str,
                        default='pipeline.config',
                        help='path to the config file')
    parser.add_argument('--output_path', required=False, type=str,
                        default='animation.mp4',
                        help='path of the saved file')
    args = parser.parse_args()
    main(args.labelmap_path,
         args.model_path,
         args.tf_record_path,
         args.config_path,
         args.output_path)


if __name__ == "__main__":
    LOGGER = utils.get_module_logger(__name__)
    main_method()
