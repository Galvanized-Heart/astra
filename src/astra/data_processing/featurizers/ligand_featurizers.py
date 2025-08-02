import torch
import numpy as np
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from astra.data_processing.featurizers.base import Featurizer


class MorganFeaturizer(Featurizer):
    """Featurizer for ligand SMILES using Morgan Fingerprints."""
    def __init__(self, radius: int, fp_size: int):
        print(f"Initializing Morgan Fingerprint generator (radius={radius}, size={fp_size})")
        self.radius = radius
        self.fp_size = fp_size
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    @property
    def name(self) -> str:
        return f"morgan_r{self.radius}_s{self.fp_size}"

    @property
    def feature_spec(self) -> Dict[str, Tuple[int, ...]]:
        """Returns the specification for the Morgan fingerprint."""
        return {"embedding": (self.fp_size,)}
    
    def featurize(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        results = {}
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp_bv = self.generator.GetFingerprint(mol)
                fp_tensor = torch.from_numpy(np.array(fp_bv, dtype=np.float32))
                results[smiles] = fp_tensor
        return results