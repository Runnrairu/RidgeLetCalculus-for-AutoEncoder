from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import mine

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






task = "Classification"


batch_x, batch_y = mnist.train.next_batch(55000) #古バッチ

a1,b1 = mine.oracle_sample(batch_x,batch_y,n,task)

x_1 = np.matmul(a1,batch_x.T)+b1

a2,b2 = mine.oracle_sample(x_1.T ,batch_y,m,task)

W_o = tf.Variable(a1)
b_o = tf.Variable(b1)
y1_o = tf.nn.relu(tf.matmul(x,W_o) + b_o)
W2_o = tf.Variable(a2)
b2_o = tf.Variable(b2)
y2_o = tf.nn.relu(tf.matmul(y1_o,W2_o) + b2_o)
W3_o = tf.Variable(tf.random_normal([m,10],stddev=0.1))
b3_o = tf.Variable(tf.zeros([10]))
y_o = tf.nn.softmax(tf.matmul(y2_o,W3_o) + b3_o)





y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

cross_entropy_o = -tf.reduce_sum(y_*tf.log(y_o))
train_step_o = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_o)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    sess.run(train_step_o)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

correct_prediction_o = tf.equal(tf.argmax(y_o,1), tf.argmax(y_,1))
accuracy_o = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print (sess.run(accuracy_o, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
