#!/bin/bash

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_dataset import Dataset

class GatedGraphConvolution(tf.keras.layers.GRU):
  def __init__(self, channels):
    super(GatedGraphConvolution, self).__init__(channels)
    self.channels = channels
  def build(self, input_shape):
    self.bias = self.add_weight(name = 'bias', shape = (1, self.channels), initializer = tf.keras.initializers.GlorotUniform(), trainable = True)
  def call(self, inputs):
    node_features, incident_node_features, context_features = inputs
    # NOTE: node_features.shape = (node_num, channels)
    # NOTE: incident_node_features.shape = (node_num, channels)
    shape = tf.shape(incident_node_features)
    hidden_states = tf.reshape(node_features, (-1, 1, channels))
    visible_states = tf.reshape(incident_node_features + self.bias, (-1, 1, channels))
    results = super(GatedGraphConvolution, self).call(visible_states, initial_state = hidden_state)
    results = tf.reshape(results, shape)
    return results
  def get_config(self):
    config = super(GatedGraphConvolution, self).get_config()
    config['channels'] = self.channels
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

def GatedGraphNeuralNetwork(channels = 256, layer_num = 4):
  graph = tf.keras.Input(type_spec = Dataset.graph_tensor_spec())
  graph = graph.merge_batch_to_components()
  graph = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(channels)(node_set[tfgnn.HIDDEN_STATE]))(graph)
  for i in range(layer_num):
    graph = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          node_input_feature = tfgnn.HIDDEN_STATE,
          edge_set_inputs = {
            "bond": tfgnn.keras.layers.SimpleConv(
              message_fn = tf.keras.layers.Identity(),
              reduce_type = "mean",
              receiver_tag = tfgnn.TARGET
            )
          },
          next_state = GatedGraphConvolution(channels)
        )
      }
    )
  results = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = 'mean', node_set_name = "atom")(graph)
  results = tf.keras.layers.Dense(1)(results)
  return tf.keras.Model(inputs = graph, outputs = results)

if __name__ == "__main__":
  adjacent = tf.sparse.expand_dims(tf.sparse.eye(10, 10), axis = 0)
  annotations = tf.random.uniform(minval = 0, maxval = 118, shape = (1,10), dtype = tf.int32)
  print(annotations)
  fe = FeatureExtractor(10)
  results = fe(adjacent, annotations)
  print(results.shape)

