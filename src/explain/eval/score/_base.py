import os
import sys
import json
import glob
from typing import Any, List, Dict, Optional

# Add the project root to Python path for debugging
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class Evaluator:
    """
    Base class for score evaluators.
    """

    def __init__(self, data_dir: str = "output", folder_name: str = "vanilla", **kwargs):
        """
        Initialize the evaluator with data from vanilla_*.json files.
        
        Args:
            data_dir: Directory containing the output files
            folder_name: Folder name within data_dir (default: "vanilla")
            **kwargs: Additional arguments
        """
        self.data_dir = data_dir
        self.folder_name = folder_name
        self.report_data = []
        self.structure_explain_data = []
        
        # Load all vanilla_*.json files
        self._load_vanilla_data()
        
        # Initialize DAG (Directed Acyclic Graph) for evaluation
        self.dag = self._initialize_dag()
        
        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _load_vanilla_data(self):
        """Load all vanilla_*.json files from the specified directory."""
        folder_path = os.path.join(self.data_dir, self.folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist")
            return
        
        # Find all vanilla_*.json files
        pattern = os.path.join(folder_path, "vanilla_*.json")
        vanilla_files = glob.glob(pattern)
        
        print(f"Found {len(vanilla_files)} vanilla files: {vanilla_files}")
        
        for file_path in vanilla_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Determine file type based on path
                if "report" in file_path:
                    self.report_data.extend(data)
                elif "structure_explain" in file_path:
                    self.structure_explain_data.extend(data)
                else:
                    # Generic data loading
                    if isinstance(data, list):
                        self.report_data.extend(data)
                    else:
                        self.report_data.append(data)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def _initialize_dag(self) -> Dict[str, Any]:
        """Initialize the DAG structure for evaluation."""
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "data_dir": self.data_dir,
                "folder_name": self.folder_name,
                "report_count": len(self.report_data),
                "structure_explain_count": len(self.structure_explain_data)
            }
        }

    def get_report_data(self) -> List[Dict[str, Any]]:
        """Get all loaded report data."""
        return self.report_data

    def get_structure_explain_data(self) -> List[Dict[str, Any]]:
        """Get all loaded structure explain data."""
        return self.structure_explain_data

    def get_perturbation_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get perturbation data by index."""
        for item in self.report_data:
            if item.get('index') == index:
                return item
        return None

    def get_structure_explain_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get structure explain data by index."""
        for item in self.structure_explain_data:
            if item.get('index') == index:
                return item
        return None

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Base evaluation method to be overridden by subclasses.
        
        Returns:
            Dictionary containing evaluation results
        """
        raise NotImplementedError("Subclasses must implement the evaluate method")

    def __repr__(self):
        return f"Evaluator(data_dir='{self.data_dir}', folder_name='{self.folder_name}', " \
               f"report_count={len(self.report_data)}, structure_explain_count={len(self.structure_explain_data)})"


