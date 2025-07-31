#
# 'binds_to(id="n1", actor="Bevacizumab", target="VEGF-A", affinity="0.15 μg/ml", via="monoclonal antibody-antigen interaction")',

from typing import Dict
import pandas as pd
from verifier_utils import get_aliases, ChEMBL_PATH, retrieve_from_bigquery


INTERACTION_TYPES = ["bind", "modulate"]

BINDING_DATA_RESOURCES = ["ChEMBL", "MatchMaker", "Boltz2"]  # to be extended

ChEMBL_DT = pd.read_csv(ChEMBL_PATH, sep=";")


def interact(entity_1, entity_2, type: str = "bind") -> Dict:
    """
    Predicts or retrieves interaction between two proteins/entities, with optional structural context.

    Entity type: protein, ligand, pathway,
    The interaction could be between Protein::Ligand, Protein::Protein, Ligand::Pathway, Protein::Pathway,
    """

    # get the standardized identifier of two entities.
    aliases_1 = get_aliases(entity_1)
    aliases_2 = get_aliases(entity_2)

    if type == "bind":
        return binds_to(entity_1, entity_2)

    elif type == "modulate":
        return modulates_to(entity_1, entity_2)


def binds_to(entity_1, entity_2):
    """
    Direct binding between protein::ligand or protein::protein.
    Aggregates predictions from multiple sources into a dictionary.
    """
    results = {}

    # ChEMBL prediction
    found, moa = get_ChEMBL_prediction(entity_1, entity_2)
    if found:
        results["ChEMBL"] = {"effect": moa}

    # MatchMaker prediction
    binding_score, _ = get_matchmaker_prediction(entity_1, entity_2)
    if binding_score is not None:
        results["MatchMaker"] = {"binding_score": binding_score}

    # Boltz2 prediction (dummy)
    binding_score, moa = get_boltz_prediction()
    if binding_score is not None or moa is not None:
        results["Boltz2"] = {"binding_score": binding_score, "effect": moa}

    return results


def get_ChEMBL_prediction(target, ligand):
    """
    Queries the ChEMBL_DT DataFrame for interactions between a specified target and ligand.

    Args:
        target (str): The ChEMBL ID of the target protein.
        ligand (str): The ChEMBL ID of the ligand molecule.

    Returns:
        tuple: A tuple (True, moa) if an interaction is found, where `moa` is a list of unique action types.
    """
    res = ChEMBL_DT.query(
        "`Target ChEMBL ID` == @target  & `Parent Molecule ChEMBL ID` == @ligand"
    )
    if res.shape[0] > 0:
        moa = res["Action Type"].unique().tolist()
        return True, moa
    return False, None


def get_matchmaker_prediction(target, ligand):
    # get the matchmaker from big query
    # rec_id, inchi_key
    res = retrieve_from_bigquery(
        sql=f"""
        SELECT mms.uniprot_name, mol.inchi_key, mms.score  
        FROM `datalake-prod-ef49c0c9.molecule_catalog_prod.mm_score` mms
        JOIN `datalake-prod-ef49c0c9.molecule_catalog_prod.molecule` mol ON mms.vendor_id = mol.vendor_id
        WHERE mms.uniprot_name = '{target}' AND mol.inchi_key = '{ligand}'
        """
    )
    assert res.shape[0] == 1
    return res.loc[0, "score"]


def get_boltz_prediction():
    # Dummy implementation, replace with actual logic
    binding_score = None
    moa = None
    return binding_score, moa


def modulates_to(entity_1, entity_2):
    """
    Functional modulation.
    """
    # Dummy implementation, replace with actual logic
    confidence = None
    effect = None
    return confidence, effect


def modulates_to(entity_1, entity_2):
    """
    Functional modulation.
    """

    return confidence, effect
