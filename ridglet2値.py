# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 04:02:11 2018

@author: 宮本来夏
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 04:02:11 2018

@author: 宮本来夏
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import mine

def double_class(y):
    b = tf.squeeze( tf.slice( y , [0,0] , [-1,1] ) )
    return tf.where( tf.greater( b , 0.5 ) , tf.ones_like(b), tf.zeros_like(b))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n = 96
m = 97



x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.random_normal([784,n], stddev=0.1))
b = tf.Variable(tf.zeros([n]))
y1 = tf.nn.relu(tf.add(tf.matmul(x,W) , b))
W2 = tf.Variable(tf.random_normal([n,m],stddev=0.1))
b2 = tf.Variable(tf.zeros([m]))
y2 = tf.nn.relu(tf.matmul(y1,W2) + b2)
W3 = tf.Variable(tf.random_normal([m,1],stddev=0.1))
b3 = tf.Variable(tf.random_normal([1]))
y = tf.nn.sigmoid(tf.matmul(y2,W3) + b3)



task = "Classification"


batch_x, batch_y = mnist.train.next_batch(55000) #古バッチ

batch_y_train_logi = mine.mnist_double(batch_y,"label")
batch_y_test_logi = mine.mnist_double(mnist.test.labels,"label")

batch_y_train = mine.mnist_double(batch_y,"OrS")



task = "Classification"

a1,b1 = mine.oracle_sample(batch_x,batch_y_train_logi,n,task)

x_1 = np.matmul(a1,batch_x.T)+b1

a2,b2_o = mine.oracle_sample(x_1.T ,batch_y_train_logi,m,task)




W_o = tf.Variable(a1.T,dtype=tf.float32)
b_o = tf.Variable(b1.T,dtype=tf.float32)

y1_o = tf.nn.relu(tf.add(tf.matmul(x,W_o) , b_o))
W2_o = tf.Variable(a2.T,dtype=tf.float32)
b2_o = tf.Variable(b2_o.T,dtype=tf.float32)
y2_o = tf.nn.relu(tf.add(tf.matmul(y1_o,W2_o) , b2_o))
W3_o = tf.Variable(tf.random_normal([m,1],stddev=0.3))
b3_o = tf.Variable(tf.random_normal([1]))

y_o = tf.nn.sigmoid(tf.add(tf.matmul(y2_o,W3_o) ,b3_o))




y_ = tf.placeholder("float", [None,1])
cross_entropy = -tf.reduce_sum(y_*tf.log(y)+(tf.ones_like(y_)-y_)*tf.log(tf.ones_like(y)-y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy,var_list=[W,b,W2,b2,W3,b3])

cross_entropy_o = -tf.reduce_sum(y_*tf.log(y_o)+(tf.ones_like(y_)-y_)*tf.log(tf.ones_like(y_o)-y_o))
train_step_o = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_o,var_list=[W_o,b_o,W2_o,b2_o,W3_o,b3_o])
correct_prediction = tf.equal(double_class(y), double_class(y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

correct_prediction_o = tf.equal(double_class(y_o), double_class(y_))
accuracy_o = tf.reduce_mean(tf.cast(correct_prediction_o, "float"))



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print (sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y_train_logi}))
print (sess.run(accuracy_o, feed_dict={x: batch_x, y_: batch_y_train_logi}))



for i in range(1000):
    
    
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_ys = mine.mnist_double(batch_ys,"label")
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    sess.run(train_step_o, feed_dict={x: batch_xs, y_: batch_ys})
    
    
print (sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y_train_logi}))
print (sess.run(accuracy_o, feed_dict={x: batch_x, y_: batch_y_train_logi}))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: batch_y_test_logi}))
print (sess.run(accuracy_o, feed_dict={x: mnist.test.images, y_: batch_y_test_logi}))
