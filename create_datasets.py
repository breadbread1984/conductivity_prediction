#!/usr/bin/python3

from absl import flags, app
from csv import reader
from rdkit import Chem
from mordred import Calculator, descriptors
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  FLAGS.DEFINE_string('input_csv', default = None, help = 'path to polymer dataset csv')
  FLAGS.DEFINE_string('output_tfrecord', default = 'dataset.tfrecord', help = 'path to output tfrecord')

class Dataset(object):
  def __init__(self):
    self.calc = Calculator(descriptors, ignore_3D = True)
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
    adjacent = tf.sparse.reorder(tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = (atom_num, atom_num)))
    row_sum = tf.sparse.reduce_sum(adjacent, axis = -1, keepdims = True) # row_sum.shape = (atom_num, 1)
    adjacent = adjacent / row_sum # normalization
    annotations = tf.stack(annotations) # annotations.shape = (atom_num)
    return adjacent, annotations
  def smiles_to_fingerprint(self, smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    feature = self.calc(molecule)
    feature = tf.constant([f for f in feature])
    return feature
  def generate_dataset(self, csv_file, tfrecord_file):
    writer = tf.io.TFRecordWriter(tfrecord_file)
    csvreader = reader(csv_file, delimiter = ',')
    next(csvreader)
    for row in csvreader:
      smiles = row[0]
      adjacent, atoms = self.smiles_to_graph(smiles)
      fingerprint = self.smiles_to_fingerprint(smiles)
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'adjacent': tf.train.Feature(bytes_list = tf.train.ByteList(value = tf.io.serialize_sparse(adjacent).numpy())),
          'atoms': tf.train.Feature(bytes_list = tf.train.ByteList(value = tf.io.serialize_tensor(atoms).numpy())),
          'feature': tf.train.Feature(bytes_list = tf.train.ByteList(value = tf.io.serialize_tensor(fingerprint).numpy()))
        }
      ))
      writer.write(trainsample.SerializeToString())
    writer.close()

if __name__ == "__main__":
  add_options()
  app.run(main)
  '''
  dataset = Dataset()
  adjacent, annotations = dataset.smiles_to_graph('c1ccccc1Cl'); print(tf.sparse.to_dense(adjacent), annotations)
  feature = dataset.smiles_to_fingerprint('c1ccccc1Cl'); print(feature)
  '''
