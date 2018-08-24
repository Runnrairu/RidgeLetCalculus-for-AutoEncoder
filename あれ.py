from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


learning_rate = 0.01 # 学習率 

batch_size = 100     # 学習1回ごと( sess.run()ごと )に訓練データをいくつ利用するか
#train_size =      # 全データの中でいくつ訓練データに回すか
step_size = 1000     # 何ステップ学習するか

# Network Parameters
n_hidden_1 = 64      # 隠れ層1のユニットの数
n_hidden_2 = 64      # 隠れ層2のユニットの数
n_input = 784          # 与える変数の数
n_classes = 10        # 分類するクラスの数 今回は生き残ったか否かなので2


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
y= multilayer_perceptron(x, weights, biases)
y_ = tf.placeholder(tf.float32, [None, 10])
# Define loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(step_size):
        
        
        
        # 訓練データから batch_size で指定した数をランダムに取得
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(optimizer,feed_dict={x : batch_xs,y_: batch_ys})
        print(sess.run(y,feed_dict={x : batch_xs,y_: batch_ys}))
        # Display logs per epoch step
        if epoch % 100 == 0:
            print("Epoch:", '%04d' % (epoch/100+1), "cost=", "{:.9f}".format(sess.run(cross_entropy,feed_dict={x : batch_xs,y_: batch_ys})))
    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



