# Import tool packages to trigger registration via __init_subclass__
from . import bio, chem  # noqa: F401
from ._base import ToolVerifier
from .esearch import LitteratureSearcher
from .evidencer import Evidencer

LITERATURE_TOOLS = dict((x.name, x) for x in [LitteratureSearcher(), Evidencer()])
BIOLOGICAL_TOOLS = dict((x.name, x) for x in [bio.DTIVerifier(), bio.SCLVerifier()])
CHEMICAL_TOOLS = dict((x.name, x) for x in [chem.ChEMBLResolver(), chem.PubChemResolver()])
REGISTERED_TOOLS = ToolVerifier._registry
