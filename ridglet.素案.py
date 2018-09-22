# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:00:57 2018

@author: 宮本来夏
"""


import tensorflow as tf
import numpy as np
import mine
import gzip
import urllib.request
import os.path
def load_mnist(filename):
    file_path ='MNIST_data'+'/' + filename
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1,784)

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

train_x = load_mnist(key_file['train_img'])
test_x = load_mnist(key_file['test_img'])
train_y = load_mnist(key_file['train_label'])
test_y = load_mnist(key_file['test_label'])




n = 96
m = 97

keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.random_normal([784,n], stddev=0.1))
b = tf.Variable(tf.zeros([n]))
y1 = tf.nn.relu(tf.add(tf.matmul(x,W) , b))
y1 = tf.nn.dropout(y1,keep_prob)
W2 = tf.Variable(tf.random_normal([n,m],stddev=0.1))
b2 = tf.Variable(tf.zeros([m]))
y2 = tf.nn.relu(tf.matmul(y1,W2) + b2)
y2 = tf.nn.dropout(y2,keep_prob)
W3 = tf.Variable(tf.random_normal([m,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
y3 = tf.matmul(y2,W3) + b3
y3 = tf.nn.dropout(y3,keep_prob)
y = tf.nn.softmax(y3)



task = "Classification"



a1,b1 = mine.oracle_Classification(train_x,train_y,n,False)

x_1 = mine.relu(np.matmul(a1,train_x.T)+b1)

a2,b2_o = mine.oracle_Classification(x_1.T ,train_y,m,False)




W_o = tf.Variable(a1.T,dtype=tf.float32)
b_o = tf.Variable(b1.T,dtype=tf.float32)
y1_o = tf.nn.relu(tf.add(tf.matmul(x,W_o) , b_o))
y1_o = tf.nn.dropout(y1_o,keep_prob)
W2_o = tf.Variable(a2.T,dtype=tf.float32)
b2_o = tf.Variable(b2_o.T,dtype=tf.float32)
y2_o = tf.nn.relu(tf.add(tf.matmul(y1_o,W2_o) , b2_o))
y2_o = tf.nn.dropout(y2_o,keep_prob)
W3_o = tf.Variable(tf.random_normal([m,10],stddev=0.1))
b3_o = tf.Variable(tf.random_normal([10]))
y3_o = tf.add(tf.matmul(y2_o,W3_o) ,b3_o)
y3_o = tf.nn.dropout(y3_o,keep_prob)
y_o = tf.nn.softmax(y3_o)





y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-15,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy,var_list=[W,b,W2,b2,W3,b3])

cross_entropy_o = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_o,1e-15,1.0)))
train_step_o = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_o,var_list=[W_o,b_o,W2_o,b2_o,W3_o,b3_o])
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

correct_prediction_o = tf.equal(tf.argmax(y_o,1), tf.argmax(y_,1))
accuracy_o = tf.reduce_mean(tf.cast(correct_prediction_o, "float"))



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print (sess.run(accuracy, feed_dict={x: train_x, y_: train_y,keep_prob:1.0}))
print (sess.run(accuracy_o, feed_dict={x: test_x, y_: test_y,keep_prob:1.0}))



for i in range(10000):
    
    
    
    
    sess.run(train_step, feed_dict={x: train_x, y_:  train_y ,keep_prob:0.5})
    sess.run(train_step_o, feed_dict={x: train_x, y_:  train_y ,keep_prob:0.5})
    
    
print (sess.run(accuracy, feed_dict={x: test_x, y_: test_y,keep_prob:1.0}))
print (sess.run(accuracy_o, feed_dict={x: test_x, y_: test_y,keep_prob:1.0}))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1.0}))
print (sess.run(accuracy_o, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1.0}))
