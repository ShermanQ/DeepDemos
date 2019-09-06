import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32)
W = tf.Variable(1.0)
b = tf.Variable(2.0)
y = W * x + b
with tf.Session as sess:
    tf.global_variables_initializer().run()
    # fetch = y.eval(feed_dict={x:3.0})
    print(sess.run(tf.get_variable("w")))
##############
print("???????????????????")
