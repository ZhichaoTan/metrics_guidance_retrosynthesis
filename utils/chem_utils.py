"""
Chemical utility functions for molecule and reaction processing.
The following implementation is based on the code from: https://github.com/jihye-roh/higherlev_retro
"""
import re

from rdkit import Chem
from rdkit.Chem import rdqueries, Descriptors

from rxnmapper import RXNMapper
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Process-level RXNMapper singleton. RXNMapper() loads a BERT transformer; instantiating it per call
# (as map_one_reaction/map_all_reactions previously did) reloads the model on every reaction and
# dominates runtime in the guided search. One lazily-created instance gives identical output.
_RXN_MAPPER = None


def _get_rxn_mapper():
    global _RXN_MAPPER
    if _RXN_MAPPER is None:
        _RXN_MAPPER = RXNMapper()
    return _RXN_MAPPER

VERBOSE = True

isCAtomsQuerier = rdqueries.AtomNumEqualsQueryAtom(6)

def has_mapping(smiles_in):
    pattern = r"\:[0-9]+\]"
    match = re.search(pattern, smiles_in)
    return bool(match)

def get_atom_maps(smiles_in):
    """Returns a set of all atom maps in input SMILES string"""
    if not smiles_in:
        return set()
    return set([int(x) for x in re.findall(r":(\d+)", smiles_in)])

def get_largest_chemical(smiles):

    """Get the molecule with the highest MW in smiles (separated by '.')"""
    
    MW = 0
    large_chemical = ''
    
    for smi in smiles.split("."):
        mol = Chem.MolFromSmiles(smi)
        if Descriptors.ExactMolWt(mol) > MW:
            large_chemical = smi
            MW = Chem.Descriptors.ExactMolWt(mol)
            
    return large_chemical


def remove_isotope_info_from_mol_in_place(mol):
    """
    Modified from https://github.com/coleygroup/react-splits/blob/main/react_splits/chem_utils.py
    Originally adapted from https://www.rdkit.org/docs/Cookbook.html#isomeric-smiles-without-isotopes
    see limitations at link about needing to canonicalize _after_.
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
       if isotope:
           atom.SetIsotope(0)
    return mol

def canonicalize_rsmi(rsmi, include_reagents=False, **otherargs):

    try: 
        reacts, reag, prods = rsmi.split('>')

        canon_reacts = sorted([canonicalize(r, **otherargs) for r in reacts.split('.')])
        canon_prods = sorted([canonicalize(p, **otherargs) for p in prods.split('.')])

        if include_reagents:
            reag = sorted(reag.split('.'))
            return '.'.join(canon_reacts)+'>'+ '.'.join(reag) + '>' + '.'.join(canon_prods)
        else:
            return '.'.join(canon_reacts)+'>>'+ '.'.join(canon_prods)

    except Exception as e:
        err_str = f"Failed to canonicalize {rsmi}"
        #warnings.warn(err_str) 
        return rsmi



def canonicalize(smiles, remove_atm_mapping=True, remove_isotope_info=False, isomericSmiles=True, **otherargs):
    
    if not remove_isotope_info and not isomericSmiles:
        smiles = smiles.replace('@', '').replace('/', '').replace('\\', '')
        isomericSmiles = True
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    if remove_atm_mapping and mol is not None:
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

    if remove_isotope_info and mol is not None:
        mol = remove_isotope_info_from_mol_in_place(mol)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles, **otherargs)


def map_one_reaction(rxn_smi):
    rxn_mapper = _get_rxn_mapper()
    reactants, products = rxn_smi.split('>>')
    can_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants.split('.')])
    can_products = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(p)) for p in products.split('.')])
    can_rsmi = f"{can_reactants}>>{can_products}"
    return rxn_mapper.get_attention_guided_atom_maps([can_rsmi])[0]["mapped_rxn"]

def map_all_reactions(route):
    rxn_mapper = _get_rxn_mapper()
    route_with_mapping = []
    for rsmi in route:
        if not has_mapping(rsmi):
            reactants, products = rsmi.split('>>')
            try:
                can_reactants = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants.split('.')])
                can_products = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(p)) for p in products.split('.')])
                can_rsmi = f"{can_reactants}>>{can_products}"
                rsmi = rxn_mapper.get_attention_guided_atom_maps([can_rsmi])[0]["mapped_rxn"]
            except:
                print(f"{rsmi} failed")
                continue
        route_with_mapping.append(rsmi)

    return route_with_mapping
