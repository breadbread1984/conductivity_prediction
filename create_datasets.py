#!/usr/bin/python3

from rdkit import Chem
import tensorflow as tf

def smiles_to_graph(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    atom_num = len(molecule.GetAtoms())
    annotations = list()
    indices = list()
    values = list()
    for atom in molecule.GetAtoms():
      idx = atom.GetIdx()
      annotation = tf.one_hot(idx, atom_num) * atom.GetAtomicNum()
      annotations.append(annotation)
      for neighbor_atom in atom.GetNeighbors():
        neighbor_idx = neighbor_atom.GetIdx()
        indices.append((idx, neighbor_idx))
        # FIXME: bond type is not shown in adjacent matrix
        #bond_type = molecule.GetBondBetweenAtoms(idx, neighbor_idx).GetBondType()
        values.append(1)
    adjacent = tf.sparse.reorder(tf.sparse.SparseTensor(indices = indices, values = values, dense_shape = (atom_num, atom_num)))
    annotations = tf.stack(annotations) # annotations.shape = (atom_num, annotation_dim)
    return adjacent, annotations

if __name__ == "__main__":
  adjacent, annotations = smiles_to_graph('c1ccccc1Cl'); print(tf.sparse.to_dense(adjacent), annotations)
