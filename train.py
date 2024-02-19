#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
import tensorflow as tf
from create_datasets import Dataset
from models import GatedGraphNeuralNetwork

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing tfrecord files')
  flags.DEFINE_integer('batch', default = 32, help = 'batch size')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('epoch', default = 200, help = 'epoch number')
  flags.DEFINE_integer('channels', default = 256, help = 'output channel of gated graph neural network')
  flags.DEFINE_integer('layers', default = 4, help = 'number of layers in gated graph neural network')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 100000, help = 'decay steps')
  flags.DEFINE_integer('save_freq', default = 100000, help = 'checkpoint save frequency')

def main(unused_argv):
  parse_func = Dataset().get_parse_function()
  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_func).prefetch(10).shuffle(10).batch(FLAGS.batch)
  valset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'testset.tfrecord')).map(parse_func).prefetch(10).shuffle(10).batch(FLAGS.batch)

  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))
  model = GatedGraphNeuralNetwork(channels = FLAGS.channels, num_layers = FLAGS.layers)
  loss = [tf.keras.losses.MeanAbsoluteError()]
  metrics = [tf.keras.metrics.MeanAbsoluteError()]
  model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  if exists(FLAGS.ckpt): model.load_weight(join(FLAGS.ckpt, 'ckpt', 'variables', 'variables'))
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'ckpt'), save_freq = FLAGS.save_freq, save_best_only = True, mode = "min")
  ]
  model.fit(trainset, epochs = FLAGS.epoch, validation_data = valset, callbacks = callbacks)

if __name__ == "__main__":
  add_options()
  app.run(main)
