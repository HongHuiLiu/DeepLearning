# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:24:23 2017

@author: lhh
"""

#####################coding=utf-8#########################################################
####################### tfrecords制作 ################################################
####################### created by tengxing on 2017.3 ####################################
####################### github：github.com/tengxing ######################################
###########################################################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 读取mnist数据。
mnist = input_data.read_data_sets("Mnist_data",dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址。
filename = "output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels), # 28x28
        'label': _int64_feature(np.argmax(labels[index])), #取出集合最大值
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
print ("TFRecord file created successful!")