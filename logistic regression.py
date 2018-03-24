from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.training import beta_decay


import tensorflow as tf
import os
from adamnc import AdamNC
from AMSGrad import AMSGrad
from ModiAdam import ModiAdamOptimizer

FLAGS = None

# Parameters
learning_rate = 0.01
training_epochs = 20000
batch_size = 100
display_step = 1000

optimizer1 = tf.train.AdamOptimizer
name1 = 'Adam'
optimizer2 = AdamNC # AdamNC
optimizer3 = AMSGrad
name3 = 'AMSGrad'
optimizer4 = tf.train.AdagradOptimizer
name4 = 'Adagrad'
optimizer5 = tf.train.GradientDescentOptimizer
name5 = 'SGD'
optimizer6 = tf.train.RMSPropOptimizer
name6 = 'RMSProp'
optimizer7 = ModiAdamOptimizer
name7 = 'ModiAdam'

opt = optimizer7
name = name7

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])


  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
  tf.summary.scalar('loss', cross_entropy)

  # t = tf.Variable(initial_value=1, trainable=False, dtype=tf.float32)

  global_step = tf.Variable(0.0, trainable=False)
  starter_learning_rate = 0.01
  end_learning_rate = 0.001
  decay_steps = 20000

  learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step=global_step,
                                              decay_steps=decay_steps, end_learning_rate=end_learning_rate,
                                              power=0.5)
  if opt == optimizer2:

    alpha=tf.constant(-0.15)
    f_old = tf.Variable(0.0, trainable=False)
    f_new = tf.Variable(1.0, trainable=False)
    beta2 = beta_decay.ratio_decay(alpha=alpha, global_step=global_step, f_old=f_old, f_new=f_new)


    add_global = global_step.assign_add(1)

    train_step = opt(learning_rate, beta2=beta2).minimize(cross_entropy)

    add_f_old = f_old.assign_add(tf.pow(global_step, -alpha))
    add_f_new = f_new.assign_add(tf.pow(tf.add(global_step, 1), -alpha))

    merged = tf.summary.merge_all()
    log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/logistic_regression')
    train_writer = tf.summary.FileWriter(log_dir + '/' + 'AdamNC-02')
    print(log_dir)

    # def get_opt_name(optimizer):
    #   st = str(optimizer)
    #
    #   return st


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for epoch in range(training_epochs + 1):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      lr, b2, summary, loss, _, _, _, _ = sess.run(
        [learning_rate, beta2, merged, cross_entropy, train_step, add_global, add_f_old, add_f_new],
        feed_dict={x: batch_xs, y_: batch_ys})
      # print(lr, b2)

      train_writer.add_summary(summary, epoch)

      if epoch % display_step == 0:
        print('training loss at epoch {}:'.format(epoch), loss)

    train_writer.close()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                          y_: mnist.test.labels}))
  else:
    train_step = opt(learning_rate).minimize(cross_entropy)

    merged = tf.summary.merge_all()
    log_dir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/logistic_regression')
    train_writer = tf.summary.FileWriter(log_dir + '/' + name)
    print(log_dir)

    # def get_opt_name(optimizer):
    #   st = str(optimizer)
    #
    #   return st


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for epoch in range(training_epochs + 1):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      lr, summary, loss, _, = sess.run(
        [learning_rate, merged, cross_entropy, train_step],
        feed_dict={x: batch_xs, y_: batch_ys})
      # print(lr, b2)

      train_writer.add_summary(summary, epoch)

      if epoch % display_step == 0:
        print('training loss at epoch {}:'.format(epoch), loss)

    train_writer.close()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                          y_: mnist.test.labels}))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
