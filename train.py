#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
import tensorflow as tf
from create_datasets import Dataset
from models import Predictor

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing tfrecord files')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('epoch', default = 400, help = 'epoch number')
  flags.DEFINE_integer('channels', default = 256, help = 'output channel of gated graph neural network')
  flags.DEFINE_integer('layers', default = 4, help = 'number of layers in gated graph neural network')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 7000, help = 'decay steps')
  flags.DEFINE_integer('save_freq', default = 700, help = 'checkpoint save frequency')

def main(unused_argv):
  parse_func = Dataset().get_parse_function()
  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_func).prefetch(10).shuffle(10).batch(1)
  testset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'testset.tfrecord')).map(parse_func).prefetch(10).shuffle(10).batch(1)

  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))
  predictor = Predictor(channels = FLAGS.channels, num_layers = FLAGS.layers)
  bc = tf.keras.losses.BinaryCrossentropy()

  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  checkpoint = tf.train.Checkpoint(model = predictor, optimizer = optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))

  log = tf.summary.create_file_writer(FLAGS.ckpt)

  for epoch in range(FLAGS.epoch):
    # train
    train_metric = tf.keras.metrics.Mean(name = 'loss')
    train_iter = iter(trainset)
    for (adjacent, atoms), label in train_iter:
      with tf.GradientTape() as tape:
        pred = predictor(adjacent, atoms)
        loss = bc(label, pred)
      train_metric.update_state(loss)
      grads = tape.gradient(loss, predictor.trainable_variables)
      optimizer.apply_gradients(zip(grads, predictor.trainable_variables))
      print('Step: #%d epoch: %d loss: %f' % (optimizer.iterations, epoch, train_metric.result()))
      if optimizer.iterations % FLAGS.save_freq == 0:
        checkpoint.save(join(FLAGS.ckpt, 'ckpt'))
        with log.as_default():
          tf.summary.scalar('loss', train_metric.result(), step = optimizer.iterations)
    # evaluation
    eval_metric = tf.keras.metrics.BinaryAccuracy()
    eval_iter = iter(testset)
    for (adjacent, atoms), label in eval_iter:
      pred = predictor(adjacent, atoms)
      eval_metric.update_state(label, pred)
      with log.as_default():
        tf.summary.scalar('binary accuracy', eval_metric.result(), step = optimizer.iterations)
      print('Step: #%d epoch: %d accuracy: %f' % (optimizer.iterations, epoch, eval_metric.result()))
  checkpoint.save(join(FLAGS.ckpt, 'ckpt'))

if __name__ == "__main__":
  add_options()
  app.run(main)
