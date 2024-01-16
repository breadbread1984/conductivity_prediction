#!/usr/bin/python3

from absl import flags, app
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.math.psd_kernels as tfk

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('num_training_points', default = 100, help = 'number of training points')
  flags.DEFINE_float('observation_var', default = .1, help = 'observation variance')
  flags.DEFINE_integer('idx_dim', default = 1, help = 'dimension of index')
  flags.DEFINE_float('learning_rate', default = .01, help = 'learning rate')
  flags.DEFINE_integer('iters', default = 1000, help = 'iteration')

def build_gp(obs_idx):
  # 全概率函数
  def gaussian_process(amplitude, length_scale, obs_samples):
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude, length_scale)
    gp = tfp.distributions.GaussianProcess(kernel = kernel,
                                         index_points = obs_idx,
                                         observation_noise_variance = obs_samples)
    return gp
  return gaussian_process

def main(unused_argv):
  # training_samples
  obs_idx = np.random.uniform(-1.,1.,(FLAGS.num_training_points, FLAGS.dim)).astype(np.float64)
  obs_samples = np.sin(3 * np.pi * obs_idx[...,0]) + np.random.normal(loc = 0., scale = np.sqrt(FLAGS.observation_var), size = (FLAGS.num_training_points,))
  # create joint distribution
  gp_join_model = tfp.distributions.JointDistributionNamed({
    'amplitude': tfp.distributions.LogNormal(loc = 0., scale = np.float64(1.)),
    'length_scale': tfp.distributions.LogNormal(loc = 0., scale = np.float64(1.)),
    'observation_noise_variance': tfp.distributions.LogNorm(loc = 0., scale = np.float64(1.)),
    'observations': build_gp(obs_idx),
  })
  # train the joint distribution given the obervation
  constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
  amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

  length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

  observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)
  
  trainable_variables = [v.trainable_variables[0] for v in
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]

  optimizer = tf.optimizers.Adam(learning_rate = FLAGS.learning_rate)
  for i in range(FLAGS.iters):
    with tf.GradientTape() as tape:
      loss = -gp_join_model.log_prob({'amplitude': amplitude_var,
                                      'length_scale': length_scale_var,
                                      'observation_noise_variance': observation_noise_variance_var,
                                      'observations': obs_samples})
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
  gp_join_model.save_weights('gp.keras')

if __name__ == "__main__":
  add_options()
  app.run(main)

