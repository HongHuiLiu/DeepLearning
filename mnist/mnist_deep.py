# tensorflow
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:47:25 2017

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

#变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#变量需要通过session初始化后，才能在session中使用
sess.run(tf.global_variables_initializer())

#类别预测
y = tf.nn.softmax(tf.matmul(x,W) + b)
#损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
