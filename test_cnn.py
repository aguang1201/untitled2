from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#params
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
logs_path = '/tmp/tensorflow_logs/test_cnn'

#network params
n_input = 784
n_classes = 10
dropout = 0.75

#tf grah input
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32, [None,n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W,b,strides = 1):
    x = tf.nn.conv2d(x,W,[1,strides,strides,1],padding="SAME")
    x = tf.nn.bias_add(x,b)
    #tf.histogram_summary("conv2d",tf.nn.relu(x))
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    #tf.histogram_summary("pool",tf.nn.max_pool(x,ksize = [1,k,k,1],strides = [1,k,k,1],padding="SAME"))
    return tf.nn.max_pool(x,ksize = [1,k,k,1],strides = [1,k,k,1],padding="SAME")

#store layers weights and bias
Weights = {
    #conv=5*5 input=1 output=32
    "wc1":tf.Variable(tf.random_normal([5,5,1,32])),
    #conv=5*5 input=32 output=64
    "wc2":tf.Variable(tf.random_normal([5,5,32,64])),
    #fully connected input=7*7*64 output=1024
    "wd1":tf.Variable(tf.random_normal([7*7*64,1024])),
    #out inout=1024 output=10(class prediction)
    "out":tf.Variable(tf.random_normal([1024,n_classes]))
}

biases = {
    "bc1":tf.Variable(tf.random_normal([32])),
    "bc2":tf.Variable(tf.random_normal([64])),
    "bd1":tf.Variable(tf.random_normal([1024])),
    "out":tf.Variable(tf.random_normal([n_classes]))
}

#create model
def conv_net(x,Weights,biases,dropout):
    x = tf.reshape(x,[-1,28,28,1])
    conv1 = conv2d(x,Weights["wc1"],biases["bc1"])
    conv1 = maxpool2d(conv1,k=2)
    conv2 = conv2d(conv1,Weights["wc2"],biases["bc2"])
    conv2 = maxpool2d(conv2,k=2)
    #fully connection layer
    fc1 = tf.reshape(conv2, [-1, Weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,Weights["wd1"]),biases["bd1"])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)

    out = tf.add(tf.matmul(fc1,Weights["out"]),biases["out"])

    return out

#constract model
with tf.name_scope("pred"):
    pred = conv_net(x,Weights,biases,keep_prob)

#define loss optimize
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#evaluate model
with tf.name_scope("accutacy"):
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

tf.scalar_summary("cost",cost)
tf.scalar_summary("accuracy",accuracy)

merged_summary = tf.merge_all_summaries()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)

    summary_writter = tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())
    step = 1
    while step * batch_size < training_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})

        _,c,summary = sess.run([optimizer,cost,merged_summary],feed_dict={x:batch_x,y:batch_y,keep_prob:1.})
        summary_writter.add_summary(summary,step*batch_size)
        if step%display_step==0:
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.})
            print("Iter" + str(step*batch_size) + ",Minibatch Loss=" + \
                  "{:.6f}".format(loss) + ",Training Accuracy=" + "{:.5f}".format(acc))
        step +=1
    print("Optimization Finished!")

    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.}))

    save_path = saver.save(sess,"my_net/save_net.ckpt")