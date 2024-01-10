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
  def __init__(self, atom_num, in_channel, out_channel,**kwargs):
    super(GatedGraphConvolution, self).__init__(**kwargs)
    self.gc = GraphConvolution()
    self.gru = tf.keras.layers.GRU(in_channel)
    self.atom_num = atom_num
    self.in_channel = in_channel
    self.out_channel = out_channel
  def call(self, adjacent, annotations):
    results = self.gc([adjacent, annotations]) # results.shape = (batch, atom_num, in_channel)
    hidden_states = tf.reshape(annotations, (-1, self.in_channel)) # hidden_states.shape = (batch * atom_num, in_channel)
    visible_states = tf.reshape(results, (-1, 1, self.in_channel)) # visible_states.shape = (batch * atom_num, 1, in_channel)
    results = self.gru(visible_states, initial_state = hidden_states) # results.shape = (batch * atom_num, in_channel)
    results = tf.reshape(results, (-1, self.atom_num, self.in_channel)) # results.shape = (batch, atom_num, in_channel)
    results = tf.keras.layers.Dense(self.out_channel, use_bias = True)(results) # results.shape = (batch, atom_num, out_channel)
    return results

class FeatureExtractor(tf.keras.Model):
  def __init__(self, atom_num, in_channel, out_channel = 32, num_layers = 4, **kwargs):
    super(FeatureExtractor, self).__init__(**kwargs)
    self.ggnns = [GatedGraphConvolution(atom_num, in_channel if i == 0 else out_channel, out_channel) for i in range(num_layers)]
  def call(self, adjacent, annotations):
    results = annotations
    for ggnn in self.ggnns:
      results = ggnn(adjacent, results)
    return results

if __name__ == "__main__":
  adjacent = tf.sparse.expand_dims(tf.sparse.eye(10, 10), axis = 0)
  annotations = tf.random.normal(shape = (1,10,100))
  fe = FeatureExtractor(10,100,200)
  results = fe(adjacent, annotations)
  print(results.shape)
