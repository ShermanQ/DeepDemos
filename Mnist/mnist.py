from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

# import tensorflow as tf
# sess = tf.InteractiveSession()
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
    """Build the mnist model up to where it may be used for inference""
    Args:
        images:Images placeholder,from inputs().
        hidden1_units:Size of first hidden layer.
        hidden2_units:Size of second hiddem layer.
    Returns:
        softmax_linear:Output tensor with the computer logits.
    """
    # hidden 1
    with tf.compat.v1.name_scope('hidden1'):
        weights = tf.Varible(
            tf.random.truncated_normal(
                [IMAGE_PIXELS,hidden1_units],
                stddev=1.0/math.sqrt(float([IMAGE_PIXELS]))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights)+biases)
    # hidden 2
    with tf.compat.v1.name_scope('hidden2'):
        weights = tf.Variable(
            tf.random.truncated_normal(
                [hidden1_units, hidden2_units],
                stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        biases=tf.Variable(tf.zeros([hidden2_units]),
                           name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights)+biases)
    # linear
    with tf.compat.v1.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.random.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev=1.0/math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights)+biases
    return logits


def loss(logits, labels):
    """Calculates the loss from logits and the labels.
    Args:
        logits:Logits tensor,float-[batch_size,NUM_CLASSES].
        labels:Labels tensor,int32-[batch_szie].

    Returns:
        loss:loss tensor of type float.

    """
    labels = tf.cast(labels, type=tf.ini64)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)


def trainning(loss, learning_rate):
    """Sets up trainning Ops.
    Create a summarizer to track the loss over time in TensorBoard.

    Create an optimizer and applies the gradients to all trainable varibales.

    The Op returned bu this function is what must passed to the 'sess.run()' call
    to cause the model to train.

    Args:
        loss:Loss tensor,from loss().
        learning_rate:The learning rate to use for gradient descent.

    Returns:
        train_op:THe Op for traning.
    """

    # Add a scalar summary for the snapshoot loss.
    tf.compat.v1.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with te given learning rate
    optimizer = tf.compat.v1.train.batch.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step=tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that mininze the loss.
    # (and also increment the global step counter)as a single training step.
    train_op = optimizer.miniminze(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """ Evaluate the quality of the logits at predicting the label.
    Args:
        logits:Logits tensor, float-[batch_size,NUM_CLASSES].
        labels:Labels tensor,int32-[batch_size,with values in the
        range[0,NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the nunber of examples(out of batch_size)
        that were predicted correctly.
    """
    correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

    return tf.reduce_sum(input_tensor=tf.cast(correct, tf.int32))
