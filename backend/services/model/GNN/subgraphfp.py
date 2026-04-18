from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors as rdDesc
from collections import defaultdict
import numpy as np
import os, pickle, hashlib

import logging

AllChem.SetPreferCoordGen(True)


def get_atom_submol_radn(mol, radius, sanitize=True):
    atoms = []
    submols = []
    #smis = []
    for atom in mol.GetAtoms():
        atoms.append(atom)
        r = radius
        while r > 0:
            try:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom.GetIdx())
                amap={}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                if sanitize:
                    Chem.SanitizeMol(
                                    submol,
                                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                                )
                #smis.append(Chem.MolToSmiles(submol))
                submols.append(submol)
                break
            except Exception as e:
                logging.error(f"在 FindAtomEnvironmentOfRadiusN 中出错: {e}", exc_info=True)
                r -= 1

    return atoms, submols #, smis

def gen_fps_from_mol(mol, nbits=256, use_morgan=True, use_macc=False, use_rdkit=False):
    # morgan
    fp = []
    if use_morgan:
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp = fp1.tolist()
    if use_macc:
        # MACCSkeys
        fp_vec = MACCSkeys.GenMACCSKeys(mol)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp.extend(fp1.tolist())
    if use_rdkit:
        fp_vec = Chem.RDKFingerprint(mol)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp.extend(fp1.tolist())

    return fp

