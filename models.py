#!/bin/bash

import tensorflow as tf

class GraphConvolution(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(GraphConvolution, self).__init__(**kwargs)
  def build(self, input_shape):
    self.bias = self.add_weight(name = 'bias', shape = (input_shape[1][-1],), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, inputs):
    # adjacent.shape = (batch, atom_num, atom_num)
    # annotations.shape = (batch, atom_num, in_channel)
    adjacent, annotations = inputs
    results = tf.sparse.sparse_dense_matmul(adjacent, annotations) # results.shape = (batch, atom_num, in_channel)
    return results

def GatedGraphConvolution(atom_num, in_channel, out_channel):
  adjacent = tf.keras.Input((atom_num, atom_num), sparse = True)
  annotations = tf.keras.Input((atom_num, in_channel))
  results = GraphConvolution()([adjacent, annotations]) # results.shape = (batch, atom_num, in_channel)
  results = tf.keras.layers.GRU(out_channel)(results)

if __name__ == "__main__":
  gc = GraphConvolution()
  adjacent = tf.random.normal(shape = (1,10,10))
  annotations = tf.random.normal(shape = (1,10,100))
  results = gc([adjacent, annotations])
  print(results.shape)
