#!/usr/bin/python3

from rdkit import Chem
from mordred import Calculator, descriptors
import tensorflow as tf

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
    print([f for f in feature])
    feature = tf.constant([f for f in feature])
    return feature

if __name__ == "__main__":
  dataset = Dataset()
  adjacent, annotations = dataset.smiles_to_graph('c1ccccc1Cl'); print(tf.sparse.to_dense(adjacent), annotations)
  feature = dataset.smiles_to_fingerprint('c1ccccc1Cl'); print(feature)
