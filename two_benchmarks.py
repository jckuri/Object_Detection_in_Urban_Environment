import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

SIZE = 10000

def benchmark(device_name):
    shape = (SIZE, SIZE)
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
    print(f'Benchmark: Matrix Multiplication {shape} x {shape}')
    print(f'Device: {device_name}.')
    startTime = datetime.now()
    with tf.device(device_name):
        random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)
        print(sum_operation)
    dt = datetime.now() - startTime
    print("Time taken:", str(dt))

def main():
    print('')
    gpus = tf.config.list_physical_devices('GPU')
    print('tf.config.list_physical_devices(\'GPU\'):', gpus)
    print('')
    if len(gpus) > 0: benchmark('gpu')
    print('')
    benchmark('cpu')
    print('')

if __name__ == "__main__":
    main()
