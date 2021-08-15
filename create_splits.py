"""
This module helps to create splits.
"""

import argparse
import glob
import os
import shutil

from utils import get_module_logger
from utils import get_dataset


def compute_real_filename(data_dir, filename):
    """
    This function computes the real filename.
    """
    index = filename.rfind('_')
    real_filename = filename[2:index] + '.tfrecord'
    if os.path.isfile(f'{data_dir}/{real_filename}'):
        return real_filename
    return None


def count_bikes(tf_record):
    """
    This function counts bikes in an tf_record.
    """
    classes = list(tf_record['groundtruth_classes'].numpy())
    return sum([c for c in classes if c == 4])


def count_bikes_in_all_files(data_dir, dataset):
    """
    This function counts bikes in all files.
    """
    all_files = os.listdir(data_dir)
    files = [f for f in all_files if not os.path.isdir(f'{data_dir}/{f}')]
    n_files = len(files)
    print('n_files:', n_files)
    bike_count = {}
    iterator = iter(dataset)
    while True:
        tf_record = next(iterator)
        n_bikes = count_bikes(tf_record)
        filename = str(tf_record['filename'].numpy())
        filename = compute_real_filename(data_dir, filename)
        if filename in bike_count:
            bike_count[filename] += n_bikes
        else:
            bike_count[filename] = n_bikes
            print("len(bike_count):", len(bike_count))
        if len(bike_count) >= n_files:
            break
    return bike_count


def split_files_with_bikes(bike_count):
    """
    This function splits files with bikes.
    """
    bike_files = []
    bikeless_files = []
    for file in bike_count.keys():
        if bike_count[file] > 0:
            bike_files.append(file)
        else:
            bikeless_files.append(file)
    return bike_files, bikeless_files


def split_files(files):
    """
    This function splits the dataset.
    The default in Matlab is 70%:15%:15% for training:validation:test.
    """
    num = len(files)
    n_train = int(0.70 * num + 0.5)
    n_val = int(0.15 * num + 0.5)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files


def copy_files(data_dir, files, destination):
    """
    This method copies files.
    """
    for file in files:
        source = f'{data_dir}/{file}'
        destiny = f'{data_dir}/{destination}/{file}'
        shutil.copyfile(source, destiny)
        print(f'The file {destiny} was copied.')


def split_and_copy_files(data_dir, files):
    """
    This method splits and copies files.
    """
    train_files, val_files, test_files = split_files(files)
    copy_files(data_dir, train_files, 'train')
    copy_files(data_dir, val_files, 'val')
    copy_files(data_dir, test_files, 'test')


def clean_dir(directory):
    """
    This method clean a directory.
    """
    files = glob.glob(f'{directory}/*')
    for file in files:
        os.remove(file)


def clean_split_dirs(data_dir):
    """
    This method cleans the split directories.
    """
    clean_dir(f'{data_dir}/train')
    clean_dir(f'{data_dir}/val')
    clean_dir(f'{data_dir}/test')


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # _TODO: Implement function
    file_pattern = f"{data_dir}/*.tfrecord"
    dataset = get_dataset(file_pattern)
    bike_count = count_bikes_in_all_files(data_dir, dataset)
    bike_files, bikeless_files = split_files_with_bikes(bike_count)
    print('RESULTS:')
    print('len(bike_files):', len(bike_files))
    print('len(bikeless_files):', len(bikeless_files))
    print('CLEANING SPLIT DIRECTORIES.')
    clean_split_dirs(data_dir)
    print('COPYING BIKE FILES...')
    split_and_copy_files(data_dir, bike_files)
    print('COPYING BIKELESS FILES...')
    split_and_copy_files(data_dir, bikeless_files)


def main():
    """
    The main method.
    """
    parser = argparse.ArgumentParser(
        description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)


if __name__ == "__main__":
    main()
