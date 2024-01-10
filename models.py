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
  def __init__(self, atom_num, in_channel,**kwargs):
    super(GatedGraphConvolution, self).__init__(**kwargs)
    self.gc = GraphConvolution()
    self.gru = tf.keras.layers.GRU(in_channel)
    self.atom_num = atom_num
    self.in_channel = in_channel
  def call(self, adjacent, annotations):
    results = self.gc([adjacent, annotations])
    hidden_states = tf.reshape(annotations, (-1, self.in_channel)) # hidden_states.shape = (batch * atom_num, in_channel)
    visible_states = tf.reshape(results, (-1, 1, self.in_channel)) # visible_states.shape = (batch * atom_num, 1, in_channel)
    results = self.gru(visible_states, initial_state = hidden_states) # results.shape = (batch * atom_num, in_channel)
    results = tf.reshape(results, (-1, self.atom_num, self.in_channel)) # results.shape = (batch, atom_num, in_channel)
    return results

if __name__ == "__main__":
  gc = GraphConvolution()
  adjacent = tf.sparse.expand_dims(tf.sparse.eye(10, 10), axis = 0)
  annotations = tf.random.normal(shape = (1,10,100))
  results = gc([adjacent, annotations])
  print(results.shape)
  ggc = GatedGraphConvolution(10,100)
  results = ggc(adjacent, annotations)
  print(results.shape)
