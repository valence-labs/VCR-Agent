from rdkit.Chem import MolFromSmiles, MolToInchiKey, SaltRemover  # noqa
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_mol(
    mol,
    disconnect_metals: bool = False,
    normalize: bool = True,
    reionize: bool = True,
    strip_salts: bool = True,
):
    r"""
    This function returns a standardized version the given molecule. It relies on the
    RDKit [`rdMolStandardize` module](https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html)
    which is largely inspired from [MolVS](https://github.com/mcs07/MolVS).
    """
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)

    if disconnect_metals:
        md = rdMolStandardize.MetalDisconnector()
        mol = md.Disconnect(mol)

    if normalize:
        mol = rdMolStandardize.Normalize(mol)

    if reionize:
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)

    if strip_salts:
        remover = SaltRemover.SaltRemover(defnData=None)
        mol = remover.StripMol(mol, sanitize=True)
    return mol


def to_inchikey(smiles: str, standardize: bool = True) -> str:
    """Compute InChIKey from SMILES.
    Args:
        smiles: SMILES string
        standardize: Whether to standardize the molecule
    Returns:
        InChIKey string
    """
    mol = standardize_mol(smiles) if standardize else MolFromSmiles(smiles)
    return MolToInchiKey(mol)


def standardize_smiles(smiles: str) -> str:
    r"""
    Apply smile standardization procedure. This is a convenient function wrapped arrounf RDKit
    smiles standardizer and tautomeric canonicalization.

    Args:
        smiles: Smiles to standardize

    Returns:
        standard_smiles: the standardized smiles
    """

    smiles = rdMolStandardize.StandardizeSmiles(smiles)
    return smiles
