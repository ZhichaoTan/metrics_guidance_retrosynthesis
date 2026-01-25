"""Graph Edit Distance scorer for reaction evaluation.
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import csv
import logging
logging.getLogger().setLevel(logging.INFO)

class GED_scorer:
    def __init__(self, reaction_smarts, product_mols=[], reactant_mols=[], use_smarts=False):
        build_mol_fun = Chem.MolFromSmiles if not use_smarts else Chem.MolFromSmarts
        if reaction_smarts and not product_mols and not reactant_mols:
            if reaction_smarts.count(">") == 1:
                reaction_smarts = reaction_smarts.replace(">", ">>")
            self.reactants_smiles, _, self.products_smiles = reaction_smarts.split(">")
            self.products = [build_mol_fun(smiles) for smiles in self.products_smiles.split(".")]
            self.reactants = [build_mol_fun(smiles) for smiles in self.reactants_smiles.split(".")]
        elif product_mols and reactant_mols:
            self.products = product_mols
            self.reactants = reactant_mols
        self.product_atoms = self.get_mapped_atoms(self.products)
        self.reactant_atoms = self.get_mapped_atoms(self.reactants)
        unmapped_atom_number = set(self.reactant_atoms.keys()) - set(self.product_atoms.keys())
        logging.debug(f"Number of unmapped atoms {unmapped_atom_number}")
        if unmapped_atom_number:
            for mol in self.reactants:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in unmapped_atom_number:
                        atom.SetAtomMapNum(0)

        self.sorted_mapped_atoms = sorted(self.product_atoms.keys())
        self.product_adjacency_matrix = self.build_adjacency_matrix(self.products, self.product_atoms)
        self.reactant_adjacency_matrix = self.build_adjacency_matrix(self.reactants, self.product_atoms)

        self.product_adjacency_matrix = self.edit_chirality(self.products, self.product_adjacency_matrix)
        self.reactant_adjacency_matrix = self.edit_chirality(self.reactants, self.reactant_adjacency_matrix)

        self.core_atom_score = self.calculate_adjacency_difference()
        self.byproduct_score = self.calculate_byproduct_difference()

    def get_mapped_atoms(self, molecules):
        mapped_atoms = {}
        for mol in molecules:
            for atom in mol.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num:
                    mapped_atoms[atom_map_num] = atom.GetIdx()
        return mapped_atoms

    def build_adjacency_matrix(self, molecules, mapped_atoms):
        num_mapped_atoms = len(mapped_atoms)
        adjacency_matrix = np.zeros((num_mapped_atoms, num_mapped_atoms))

        for mol in molecules:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtom()
                j = bond.GetEndAtom()
                i_map = i.GetAtomMapNum()
                j_map = j.GetAtomMapNum()
                if i_map in mapped_atoms and j_map in mapped_atoms:
                    idx_i = self.sorted_mapped_atoms.index(i_map)
                    idx_j = self.sorted_mapped_atoms.index(j_map)
                    bond_order = bond.GetBondTypeAsDouble()
                    adjacency_matrix[idx_i, idx_j] = bond_order
                    adjacency_matrix[idx_j, idx_i] = bond_order
        return adjacency_matrix

    def calculate_adjacency_difference(self):
        difference_matrix = np.abs(self.product_adjacency_matrix - self.reactant_adjacency_matrix)
        total_difference = np.sum(difference_matrix) / 2
        self.chirality_score = np.sum(np.diagonal(difference_matrix))/2

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(difference_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        # plt.title("Difference Matrix Heatmap")
        # plt.savefig("heatmap.png")

        return total_difference

    def calculate_byproduct_difference(self):
        mapped_bonds = 0
        for react in self.reactants:
            for bond in react.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                bond_order = bond.GetBondTypeAsDouble()

                if begin_atom.GetAtomMapNum() != 0 and end_atom.GetAtomMapNum() == 0:
                    mapped_bonds += bond_order
                elif begin_atom.GetAtomMapNum() == 0 and end_atom.GetAtomMapNum() != 0:
                    mapped_bonds += bond_order
        return mapped_bonds

    def edit_chirality(self, molecules, matrix):

        for mol in molecules:
            for atom in mol.GetAtoms():
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num in self.sorted_mapped_atoms:
                    chirality = int(atom.GetChiralTag())
                    idx = self.sorted_mapped_atoms.index(atom_map_num)
                    matrix[idx, idx] = 1 if chirality else 0
                    matrix[idx, idx] = matrix[idx, idx] * 2
        return matrix

    def calculate_score(self):
        final_score = self.core_atom_score + self.byproduct_score
        return final_score

if __name__ == '__main__':
    example_route = ['[CH4:1].[CH:2](=[CH2:3])[CH2:5][CH:6]=[O:7].[OH2:4]>>[CH3:1][C@H:2]([CH2:3][OH:4])[CH2:5][CH:6]=[O:7]',
                 "[C:11](=[O:12])([CH2:13][CH2:14][CH:19]=[O:20])[OH:18].[CH3:1][C@H:2]([CH2:3][OH:4])[CH2:16][CH:6]=[O:5].[NH2:10][CH3:15]>[CH3:1][C@H:2]([CH2:3][OH:4])[C@H:16]([CH:6]=[O:5])[C@H:15]([NH2:10])[CH2:14][CH2:13][C:11](=[O:12])[OH:18]",
                 '[CH2:1]=[CH:2][CH3:3].[CH:4](=[O:5])[C@@H:6]([C@H:7]([CH3:8])[CH2:9][OH:10])[C@H:11]([NH2:12])[CH2:13][CH2:14][C:15](=[O:16])[OH:17]>>[CH2:1]=[CH:2][CH2:3][C@H:4]([OH:5])[C@@H:6]([C@H:7]([CH3:8])[CH2:9][OH:10])[C@H:11]([NH2:12])[CH2:13][CH2:14][C:15](=[O:16])[OH:17]',
                 '[CH2:1]=[CH:2][CH2:3][C@H:4]([OH:5])[C@@H:6]([C@H:7]([CH3:8])[CH2:9][OH:10])[C@@H:11]([CH2:12][CH2:13][C:14](=[O:15])[OH:17])[NH2:16]>>[CH2:1]=[CH:2][CH2:3][C@H:4]([OH:5])[C@@H:6]([C@H:7]([CH3:8])[CH2:9][OH:10])[C@@H:11]1[CH2:12][CH2:13][C:14](=[O:15])[NH:16]1.[OH2:17]',
                 '[CH2:1]=[CH:2][CH2:3][C@H:4]([OH:5])[C@@H:10]([C@@H:8]([CH2:6][OH:7])[CH3:9])[C@@H:11]1[CH2:12][CH2:13][C:14](=[O:15])[NH:16]1>>[CH2:1]=[CH:2][CH2:3][C@H:4]1[O:5][C:6](=[O:7])[C@@H:8]([CH3:9])[C@@H:10]1[C@@H:11]1[CH2:12][CH2:13][C:14](=[O:15])[NH:16]1',
                 '[BrH:10].[CH3:1][C@@H:2]1[C:3](=[O:4])[O:5][C@H:6]([CH2:7][CH:8]=[CH2:9])[C@H:11]1[C@@H:12]1[CH2:13][CH2:14][C:15](=[O:16])[NH:17]1>>[CH3:1][C@@H:2]1[C:3](=[O:4])[O:5][C@H:6]([CH2:7][CH2:8][CH2:9][Br:10])[C@H:11]1[C@@H:12]1[CH2:13][CH2:14][C:15](=[O:16])[NH:17]1',
                 '[Br:1][CH2:10][CH2:9][CH2:8][C@H:7]1[O:6][C:4](=[O:5])[C@@H:3]([CH3:2])[C@@H:17]1[C@H:16]1[NH:11][C:12](=[O:13])[CH2:14][CH2:15]1>>[BrH:1].[CH3:2][C@@H:3]1[C:4](=[O:5])[O:6][C@H:7]2[CH2:8][CH2:9][CH2:10][N:11]3[C:12](=[O:13])[CH2:14][CH2:15][C@@H:16]3[C@@H:17]12']
    for reaction_smiles in example_route:
        scorer = GED_scorer(reaction_smiles)
        score = scorer.calculate_score()
        print(score)
