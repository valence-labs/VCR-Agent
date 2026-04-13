import json
import os

from lxml import etree


class DrugBankSearcher:
    def __init__(self, xml_file_path: str = "/rxrx/data/user/lu.zhu/outgoing/data/drugbank20231211/full_database.xml"):
        """
        Initialize the DrugBank searcher with the XML file path.
        Uses lazy loading and indexing for memory efficiency.

        Args:
            xml_file_path: Path to the DrugBank XML file
        """
        self.xml_file_path = xml_file_path
        self._name_to_id_index = None
        self._synonym_to_id_index = None

    def _build_index(self):
        """Build name and synonym indices for fast lookup."""
        if os.path.exists('data/drugbank/name_to_id_index.json'):
            self._name_to_id_index = json.load(open('data/drugbank/name_to_id_index.json'))
        if os.path.exists('data/drugbank/synonym_to_id_index.json'):
            self._synonym_to_id_index = json.load(open('data/drugbank/synonym_to_id_index.json'))

        if self._name_to_id_index is not None:
            return

        print("Building DrugBank search indices...")
        self._name_to_id_index = {}
        self._synonym_to_id_index = {}

        # Use iterparse for memory-efficient XML parsing
        context = etree.iterparse(self.xml_file_path, events=('end',), tag='{*}drug')

        for _event, drug in context:
            try:
                # Get drug ID
                drug_id_elem = drug.find('.//{*}drugbank-id')
                if drug_id_elem is None or drug_id_elem.text is None:
                    continue
                drug_id = drug_id_elem.text

                # Get drug name
                name_elem = drug.find('.//{*}name')
                if name_elem is not None and name_elem.text is not None:
                    self._name_to_id_index[name_elem.text.lower()] = drug_id

                # Get synonyms
                synonyms_elem = drug.find('.//{*}synonyms')
                if synonyms_elem is not None:
                    for synonym in synonyms_elem.findall('.//{*}synonym'):
                        if synonym.text is not None:
                            self._synonym_to_id_index[synonym.text.lower()] = drug_id

            except Exception as e:
                print(f"Warning: Error processing drug element: {e}")
                continue

        print(f"Built indices with {len(self._name_to_id_index)} names and {len(self._synonym_to_id_index)} synonyms")

    def search_by_name(self, name: str) -> str | None:
        """
        Search for a drug by name using pre-built indices.

        Args:
            name: Drug name

        Returns:
            DrugBank ID
        """
        if self._name_to_id_index is None:
            self._build_index()

        name_lower = name.lower()

        # Check exact name match
        if name_lower in self._name_to_id_index:
            return self._name_to_id_index[name_lower]

        # Check synonym match
        if name_lower in self._synonym_to_id_index:
            return self._synonym_to_id_index[name_lower]

        return None
