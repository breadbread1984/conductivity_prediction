#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
from rdkit import Chem
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to polymer dataset csv')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

class Dataset(object):
  def __init__(self):
    pass
  def smiles_to_graph(self, smiles: str, label: float):
    molecule = Chem.MolFromSmiles(smiles)
    atom_num = len(molecule.GetAtoms())
    idices = list()
    nodes = list()
    edges = list()
    for atom in molecule.GetAtoms():
      idx = atom.GetIdx()
      indices.append(idx)
      nodes.append(atom.GetAtomicNum())
      for neighbor_atom in atom.GetNeighbors():
        neighbor_idx = neighbor_atom.GetIdx()
        bond = molecule.GetBondBetweenAtoms(idx, neighbor_idx)
        edges.append((idx, neighbor_idx, bond.GetBondType()))
    indices = np.array(indices)
    nodes = np.array(nodes)
    edges = np.array(edges)
    sidx = np.argsort(indices)
    nodes = nodes[sidx]
    graph = tfgnn.GraphTensor.from_pieces(
      node_sets = {
        "atom": tfgnn.NodeSet.from_fields(
          sizes = tf.constant([nodes.shape[0]]),
          features = {
            tfgnn.HIDDEN_STATE: tf.one_hot(nodes, 118),
          }
        )
      },
      edge_sets = {
        "bond": tfgnn.EdgeSet.from_fields(
          sizes = tf.constant([edges.shape[0]]),
          adjacency = tfgnn.Adjacency.from_indices(
            source = ("atom", edges[:,0]),
            target = ("atom", edges[:,1])
          ),
          features = {
            tfgnn.HIDDEN_STATE: tf.one_hot(edges[:,2], 22),
          }
        )
      },
      context = tfgnn.Context.from_fields(
        features = {
          "label": tf.constant([label,], dtype = tf.float32)
        }
      )
    )
    return graph
  @staticmethod
  def graph_tensor_spec():
    spec = tfgnn.GraphTensorSpec.from_piece_specs(
      node_sets_spec = {
        "atom": tfgnn.NodeSetSpec.from_field_specs(
          features_spec = {
            tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 118), tf.float32)
          },
          sizes_spec = tf.TensorSpec((1,), tf.int32)
        )
      },
      edge_sets_spec = {
        "bond": tfgnn.EdgeSetSpec.from_field_specs(
          features_spec = {
            tfgnn.HIDDEN_STATE: tf.TensorSpec((None, 22), tf.float32)
          },
          sizes_spec = tf.TensorSpec((1,), tf.int32),
          adjacency_spec = tfgnn.AdjacencySpec.from_incident_node_sets("atom", "atom")
        )
      },
      context_spec = tfgnn.ContextSpec.from_field_specs(
        features_spec = {
          "label": tf.TensorSpec(shape = (1,), dtype = tf.float32)
        }
      )
    )
    return spec
  def generate_dataset(self, csv_file, output_dir):
    if exists(output_dir): rmtree(output_dir)
    mkdir(output_dir)
    samples = list()
    csv = open(csv_file, 'r')
    for line, row in enumerate(csv.readlines()):
      if line == 0: continue
      smiles, label = row.split(',')
      samples.append((smiles, label))
    is_train = np.random.multinomial(1, [9/10,1/10], size = len(samples))[:,0].astype(np.bool_)
    samples = np.array(samples)
    trainset = samples[is_train].tolist()
    valset = samples[np.logical_not(is_train)].tolist()
    self.generate_tfrecord(trainset, join(output_dir, 'trainset.tfrecord'))
    self.generate_tfrecord(valset, join(output_dir, 'testset.tfrecord'))
    csv.close()
  def generate_tfrecord(self, samples, tfrecord_file):
    writer = tf.io.TFRecordWriter(tfrecord_file)
    for line, (smiles, label) in enumerate(samples):
      graph = self.smiles_to_graph(smiles, float(label))
      example = tfgnn.write_example(graph)
      writer.write(example.SerializeToString())
    writer.close()
  def get_parse_function(self,):
    def parse_function(serialized_example):
      graph = tfgnn.parse_single_example(
        self.graph_tensor_spec(),
        serialized_example,
        validate = True)
      context_features = graph.context.get_features_dict()
      label = context_features.pop('label')
      graph = graph.replace_features(context = context_features)
      return graph, label
    return parse_function

def main(unused_argv):
  dataset = Dataset()
  dataset.generate_dataset(FLAGS.input_csv, FLAGS.output_dir)

if __name__ == "__main__":
  add_options()
  app.run(main)
  '''
  dataset = Dataset()
  adjacent, annotations = dataset.smiles_to_graph('c1ccccc1Cl'); print(tf.sparse.to_dense(adjacent), annotations)
  feature = dataset.smiles_to_fingerprint('c1ccccc1Cl'); print(feature)
  '''
