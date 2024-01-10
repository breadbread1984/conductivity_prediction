#!/bin/bash

import tensorflow as tf

class GraphConvolution(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GraphConvolution, self).__init__(**kwargs)
  def build(self, input_shape):
    self.bias = self.add_weight(name = 'bias', shape = (1,1,input_shape[1][-1]), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, inputs):
    # adjacent.shape = (batch, atom_num, atom_num)
    # annotations.shape = (batch, atom_num, in_channel)
    adjacent, annotations = inputs
    results = list()
    # NOTE: sparse_dense_matmul doesn't support matrix with batch dimension
    for i in range(tf.shape(adjacent)[0]):
      adj = tf.sparse.slice(adjacent, [i,0,0], [1,tf.shape(adjacent)[1],tf.shape(adjacent)[2]])
      adj = tf.sparse.reshape(adj, [tf.shape(adjacent)[1], tf.shape(adjacent)[2]])
      results.append(tf.sparse.sparse_dense_matmul(adj, annotations[i])) # results.shape = (batch, atom_num, in_channel)
    results = tf.stack(results, axis = 0)
    results = results + self.bias
    return results

class GatedGraphConvolution(tf.keras.Model):
  def __init__(self, channels, **kwargs):
    super(GatedGraphConvolution, self).__init__(**kwargs)
    self.gc = GraphConvolution()
    self.gru = tf.keras.layers.GRU(channels)
    self.channels = channels
  def call(self, adjacent, annotations):
    results = self.gc([adjacent, annotations]) # results.shape = (batch, atom_num, channels)
    shape = tf.shape(results)
    hidden_states = tf.reshape(annotations, (-1, self.channels)) # hidden_states.shape = (batch * atom_num, channels)
    visible_states = tf.reshape(results, (-1, 1, self.channels)) # visible_states.shape = (batch * atom_num, 1, channels)
    results = self.gru(visible_states, initial_state = hidden_states) # results.shape = (batch * atom_num, channels)
    results = tf.reshape(results, shape) # results.shape = (batch, atom_num, channels)
    return results

class FeatureExtractor(tf.keras.Model):
  def __init__(self, channels = 32, num_layers = 4, **kwargs):
    super(FeatureExtractor, self).__init__(**kwargs)
    self.embed = tf.keras.layers.Embedding(118, channels)
    self.ggnns = [GatedGraphConvolution(channels) for i in range(num_layers)]
  def call(self, adjacent, annotations):
    results = self.embed(annotations) # results.shape = (batch, atom_num, 32)
    for ggnn in self.ggnns:
      results = ggnn(adjacent, results)
    return results

class FingerPrint(tf.keras.Model):
  def __init__(self, channels = 32, num_layers = 4, **kwargs):
    super(FingerPrint, self).__init__(**kwargs)
    self.fe = FeatureExtractor(channels, num_layers, **kwargs)
    self.dense1 = tf.keras.layers.Dense(100, use_bias = True, activation = tf.keras.activations.relu)
    self.dense2 = tf.keras.layers.Dense(1613, use_bias = False)
    self.dropout = tf.keras.layers.Dropout(rate = 0.1)
  def call(self, adjacent, annotations):
    results = self.fe(adjacent, annotations)
    results = self.dense1(results)
    results = self.dense2(results)
    results = self.dropout(results)
    return results

if __name__ == "__main__":
  adjacent = tf.sparse.expand_dims(tf.sparse.eye(10, 10), axis = 0)
  annotations = tf.random.uniform(minval = 0, maxval = 118, shape = (1,10), dtype = tf.int32)
  print(annotations)
  fe = FeatureExtractor(10)
  results = fe(adjacent, annotations)
  print(results.shape)
  fp = FingerPrint(10)
  results = fp(adjacent, annotations)
  print(results.shape)
