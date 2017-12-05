# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:10:51 2017

@author: lhh
"""
#自动下载和导入MNIST数据集，它会自动创建一个‘MNIST_data’的目录来存储数据
import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()
#占位符
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])


#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化
#我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
#我们的池化用简单传统的2*2大小的模板做max pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
#第一层卷积  28*28-28*28
#5*5的patch中算出32个特征。卷积的权重张量形状是[5，5，1，32],前两个维度是patch的大小
#接着是输入的通道数据，最后是输出的通道数目  
W_conv1 = weight_variable([5, 5, 1, 32])
#第一层池化  28*28-14*14
b_conv1 = bias_variable([32])

#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
#最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#密集连接层
#图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
#我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout：防止过拟合-把噪声数据的特征也学习到了
#解决办法:跳过一定的神经元
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#变量需要通过session初始化后，才能在session中使用
sess.run(tf.global_variables_initializer())

#步骤
#1、在外部构建图
#2、在tensorflow内部运算图

for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

















