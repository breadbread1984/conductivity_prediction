#!/usr/bin/python3

from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np

class Smiles2Feature(object):
  def __init__(self,):
    self.calc = Calculator(descriptors, ignore_3D = False)
  def extract(self, smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    feature = self.calc(molecule)
    print('---------------------', feature[0])
    feature = np.array([f for f in feature]).astype(np.float32)
    return feature

if __name__ == "__main__":
  extractor = Smiles2Feature()
  feature1 = extractor.extract('c1ccccc1Cl'); print(feature1)
  feature2 = extractor.extract('c1ccccc1O'); print(feature2)
  feature3 = extractor.extract('c1ccccc1N'); print(feature3)
