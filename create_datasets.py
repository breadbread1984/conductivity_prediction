#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import join, exists
from rdkit import Chem
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_csv', default = None, help = 'path to polymer dataset csv')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

class Dataset(object):
  def __init__(self):
    pass
  def smiles_to_graph(self, smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    atom_num = len(molecule.GetAtoms())
    annotations = list()
    indices = list()
    values = list()
    for atom in molecule.GetAtoms():
      idx = atom.GetIdx()
      annotations.append(atom.GetAtomicNum())
      for neighbor_atom in atom.GetNeighbors():
        neighbor_idx = neighbor_atom.GetIdx()
        indices.append((idx, neighbor_idx))
        # FIXME: bond type is not shown in adjacent matrix
        #bond_type = molecule.GetBondBetweenAtoms(idx, neighbor_idx).GetBondType()
        values.append(1)
    adjacent = tf.cast(tf.sparse.reorder(tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = (atom_num, atom_num))), dtype = tf.float32)
    row_sum = tf.sparse.reduce_sum(adjacent, axis = -1, keepdims = True) # row_sum.shape = (atom_num, 1)
    adjacent = adjacent / row_sum # normalization
    annotations = tf.cast(tf.stack(annotations), dtype = tf.int32) # annotations.shape = (atom_num)
    return adjacent, annotations
  def generate_dataset(self, csv_file, output_dir):
    if exists(output_dir): rmtree(output_dir)
    mkdir(output_dir)
    samples = list()
    csv = open(csv_file, 'r')
    for line, row in enumerate(csv.readlines()):
      if line == 0: continue
      smiles, label = row.split(',')
      samples.append((smiles, label))
    is_train = np.random.multinomial(1, [9/10,1/10], size = len(samples))[:,0].astype(np.bool)
    samples = np.array(samples)
    trainset = samples[is_train].tolist()
    valset = samples[is_train].tolist()
    self.generate_tfrecord(trainset, join(output_dir, 'trainset.tfrecord'))
    self.generate_tfrecord(valset, join(output_dir, 'testset.tfrecord'))
  def generate_tfrecord(self, samples, tfrecord_file):
    writer = tf.io.TFRecordWriter(tfrecord_file)
    for line, (smiles, label) in enumerate(samples):
      adjacent, atoms = self.smiles_to_graph(smiles)
      label = float(label)
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'adjacent': tf.train.Feature(bytes_list = tf.train.BytesList(value = tf.io.serialize_sparse(adjacent).numpy())),
          'atoms': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(atoms).numpy()])),
          'atom_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [atoms.shape[0]])),
          'label': tf.train.Feature(int64_list = tf.train.FloatList(value = [label,])),
        }
      ))
      writer.write(trainsample.SerializeToString())
    csv.close()
    writer.close()
  def get_parse_function(self,):
    def parse_function(serialized_example):
      feature = tf.io.parse_single_example(
        serialized_example,
        features = {
          'adjacent': tf.io.FixedLenFeature((1,3), dtype = tf.string),
          'atoms': tf.io.FixedLenFeature((), dtype = tf.string),
          'atom_num': tf.io.FixedLenFeature((), dtype = tf.int64),
          'label': tf.io.FixedLenFeature((), dtype = tf.float64),
        })
      adjacent = tf.io.deserialize_many_sparse(feature['adjacent'], dtype = tf.float32)
      adjacent = tf.sparse.reshape(adjacent, (tf.shape(adjacent)[1], tf.shape(adjacent)[2]))
      atoms = tf.io.parse_tensor(feature['atoms'], out_type = tf.int32)
      atom_num = tf.cast(feature['atom_num'], dtype = tf.int32)
      label = tf.cast(feature['label'], dtype = tf.float32)
      atoms = tf.reshape(atoms, (atom_num,))
      return (adjacent, atoms), label
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
