# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:00:57 2018

@author: 宮本来夏
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n = 96 #ノード数
m = 96

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.random_normal([784,n], stddev=0.1))
b = tf.Variable(tf.zeros([n]))
y1 = tf.nn.relu(tf.matmul(x,W) + b)
W2 = tf.Variable(tf.random_normal([n,m],stddev=0.1))
b2 = tf.Variable(tf.zeros([m]))
y2 = tf.nn.relu(tf.matmul(y1,W2) + b2)
W3 = tf.Variable(tf.random_normal([m,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(y2,W3) + b3)



batch_xs, batch_ys = mnist.train.next_batch(55000)
print(batch_xs)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
    
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

