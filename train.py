#!/usr/bin/python3

from absl import flags, app
from os.path import join
import tensorflow as tf
from create_datasets import Dataset
from models import FingerPrint

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset file')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('channels', default = 32, help = 'output channel of gated graph neural network')
  flags.DEFINE_integer('layers', default = 4, help = 'number of layers in gated graph neural network')
  flags.DEFINE_integer('train_step', default = 1000000, help = 'number of training step')
  flags.DEFINE_integer('display_interval', default = 100, help = 'steps for one tensorboard update')
  flags.DEFINE_integer('ckpt_interval', default = 1000, help = 'steps for one ckpt')

def main(unused_argv):
  parse_func = Dataset().get_parse_function()
  dataset = tf.data.TFRecordDataset(FLAGS.dataset).repeat().map(parse_func).prefetch(50).shuffle(50).batch(1)
  dataset_iter = iter(dataset)
  model = FingerPrint(channels = FLAGS.channels, num_layers = FLAGS.layers)
  optimizer = tf.keras.optimizer.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = FLAGS.train_step / 10, decay_rate = 0.9))
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer)
  train_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32)
  log = tf.summary.create_file_writer(FLAGS.ckpt)
  for step in range(FLAGS.train_step):
    with tf.GradientTape() as tape:
      (adjacent, atoms), feature = next(dataset)
      pred = model(adjacent, atoms)
      loss = tf.keras.losses.MeanAbsoluteError()(feature, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradient(zip(grads, model.trainable_variables))
    train_loss.update_state(feature, pred)
    if tf.equal(optimizer.iterations % FLAGS.display_interval):
      with log.as_default():
        tf.summary.scalar('loss', train_loss.result(), step = optimizer.iterations)
      print('Step #%d Train loss: %.6f' % train_loss.result())
      train_loss.reset_states()
    if tf.equal(optimizer.iterations % FLAGS.ckpt_interval):
      model.save_weights(join(FLAGS.ckpt, 'ckpt_%d.h5' % optimizer.iterations))

if __name__ == "__main__":
  add_options()
  app.run(main)
