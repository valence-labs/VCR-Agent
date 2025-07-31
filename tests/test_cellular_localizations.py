import pytest
import unittest
from unittest.mock import patch, Mock
from conf import test_scl_data
from explain.verifiers.cellular_localization import (
    scl_from_uniprot,
    scl_from_HPA,
    extract_translocation_details_with_gemini,
)
import requests


class TestCelluarLocalization(unittest.TestCase):
    def test_scl_from_uniprot_success(self):
        result, _ = scl_from_uniprot("P05067")
        self.assertEqual(
            sorted(result),
            sorted(
                [
                    "Cell membrane",
                    "Membrane",
                    "Perikaryon",
                    "Cell projection, growth cone",
                    "Membrane, clathrin-coated pit",
                    "Early endosome",
                    "Cytoplasmic vesicle",
                ]
            ),
        )

    def test_scl_from_uniprot_failure(self):
        result, _ = scl_from_uniprot("ENSG00000141510")
        self.assertEqual(result, [])

    def test_scl_from_hpa_success(self):
        result = scl_from_HPA("ENSG00000141510")
        self.assertEqual(sorted(result), sorted(["Nucleoplasm", "Vesicles", "Cytosol"]))

    def test_translocation_with_gemini(self):
        scl_text = "In response to EGF, translocated from the cell membrane to the nucleus via Golgi and ER (PubMed:17909029, PubMed:20674546). Endocytosed upon activation by ligand (PubMed:17182860, PubMed:17909029, PubMed:27153536, PubMed:2790960)."
        extracted = extract_translocation_details_with_gemini(scl_text)
        expected = [{ "from": "cell membrane", "to": "nucleus", "is_phosphorylated": None}]
        self.assertListEqual(extracted, expected)

        scl_text = "Colocalized with GPER1 in the nucleus of estrogen agonist-induced cancer-associated fibroblasts (CAF) (PubMed:20551055)"
        extracted = extract_translocation_details_with_gemini(scl_text)
        expected = []
        self.assertListEqual(extracted, expected)


        scl_text = "Translocated into the nucleus upon tyrosine phosphorylation and dimerization, in response to IFN-gamma and signaling by activated FGFR1, FGFR2, FGFR3 or FGFR4 (PubMed:15322115)."
        extracted = extract_translocation_details_with_gemini(scl_text)
        expected = [{"from": None, "to": "nucleus", "is_phosphorylated": True}]
        self.assertListEqual(extracted, expected)