"""
Chemical utility functions for molecule and reaction processing.
"""
from rdkit import Chem
from rdkit.Chem import AllChem

def has_mapping(rsmi):
    """
    Check if a reaction SMILES has atom mapping.

    Args:
        rsmi: Reaction SMILES string

    Returns:
        bool: True if reaction has atom mapping, False otherwise
    """
    if not rsmi or ">>" not in rsmi:
        return False

    try:
        reactants, products = rsmi.split(">>")
        # Check if any atom has a mapping number (non-zero)
        for smiles in reactants.split(".") + products.split("."):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        return True
    except:
        pass
    return False

def get_atom_maps(tagged_smiles):
    """
    Extract atom mapping numbers from tagged SMILES.

    Args:
        tagged_smiles: SMILES string with atom mapping

    Returns:
        list: List of atom mapping numbers
    """
    mol = Chem.MolFromSmiles(tagged_smiles)
    if not mol:
        return []

    atom_maps = []
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            atom_maps.append(map_num)
    return atom_maps

def get_largest_chemical(smiles):
    """
    Get the largest chemical (by number of atoms) from a SMILES string.

    Args:
        smiles: SMILES string (may contain multiple molecules separated by '.')

    Returns:
        str: SMILES of the largest molecule
    """
    if not smiles:
        return ""

    molecules = smiles.split(".")
    if len(molecules) == 1:
        return molecules[0]

    largest = ""
    max_atoms = 0

    for mol_smiles in molecules:
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol:
            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms
                largest = mol_smiles

    return largest if largest else molecules[0]

def canonicalize(smiles):
    """
    Canonicalize a SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        str: Canonical SMILES string
    """
    if not smiles:
        return ""

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return smiles

def canonicalize_rsmi(rsmi):
    """
    Canonicalize a reaction SMILES string.

    Args:
        rsmi: Reaction SMILES string (format: reactants>>products or reactants>reagents>products)

    Returns:
        str: Canonical reaction SMILES
    """
    if not rsmi:
        return ""

    try:
        parts = rsmi.split(">")
        if len(parts) == 2:
            # reactants>>products
            reactants, products = parts
            can_reactants = ".".join([canonicalize(r) for r in reactants.split(".") if r])
            can_products = ".".join([canonicalize(p) for p in products.split(".") if p])
            return f"{can_reactants}>>{can_products}"
        elif len(parts) == 3:
            # reactants>reagents>products
            reactants, reagents, products = parts
            can_reactants = ".".join([canonicalize(r) for r in reactants.split(".") if r])
            can_reagents = ".".join([canonicalize(r) for r in reagents.split(".") if r])
            can_products = ".".join([canonicalize(p) for p in products.split(".") if p])
            return f"{can_reactants}>{can_reagents}>{can_products}"
    except:
        pass
    return rsmi

def canonicalize_route(route):
    """
    Canonicalize a list of reaction SMILES.

    Args:
        route: List of reaction SMILES strings

    Returns:
        list: List of canonicalized reaction SMILES
    """
    return [canonicalize_rsmi(rsmi) for rsmi in route]
