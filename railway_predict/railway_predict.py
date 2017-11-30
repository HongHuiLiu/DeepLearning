#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#——————————————————导入数据——————————————————————
f=open('railway.csv')  
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['pass'])   #获取最高价序列

#以折线图展示data
plt.figure()
plt.plot(data)
plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  #标准化
normalize_data=normalize_data[:,np.newaxis]       #增加维度      

# normalize  
normalized_data = (data - np.mean(data)) / np.std(data)  
   
seq_size = 3  
train_x, train_y = [], []  
for i in range(len(normalized_data) - seq_size - 1):  
    train_x.append(np.expand_dims(normalized_data[i : i + seq_size], axis=1).tolist())  
    train_y.append(normalized_data[i + 1 : i + seq_size + 1].tolist())  
   
input_dim = 1  
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])  
Y = tf.placeholder(tf.float32, [None, seq_size])  
   
# regression  
def ass_rnn(hidden_layer_size=6):  
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')  
    b = tf.Variable(tf.random_normal([1]), name='b')  
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)  
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)  
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])  
    out = tf.matmul(outputs, W_repeated) + b  
    out = tf.squeeze(out)  
    return out  
   
def train_rnn():  
    out = ass_rnn()  
   
    loss = tf.reduce_mean(tf.square(out - Y))  
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)  
   
    saver = tf.train.Saver(tf.global_variables())  
    with tf.Session() as sess:  
        tf.get_variable_scope().reuse_variables()  
        sess.run(tf.global_variables_initializer())  
   
        for step in range(10000):  
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})  
            if step % 10 == 0:  
                # 用测试数据评估loss  
                print(step, loss_)  
        print("保存模型: ", saver.save(sess, 'model_save\\modle.ckpt'))  
   
#train_rnn()  
   
def prediction(): 
    #with tf.variable_scope("sec_lstm",reuse=True):
    out = ass_rnn()  
   
    saver = tf.train.Saver(tf.global_variables())  
    with tf.Session() as sess:  
        tf.get_variable_scope().reuse_variables()  
        saver.restore(sess, 'model_save\\modle.ckpt')  
          
        prev_seq = train_x[-1]  
        predict = []  
        for i in range(12):  
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})  
            predict.append(next_seq[-1])  
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))  
   
        plt.figure()  
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')  
        plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')  
        plt.show()  
   
prediction()  