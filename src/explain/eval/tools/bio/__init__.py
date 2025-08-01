# Simply importing the tool classes is now enough to trigger their registration
# via the __init_subclass__ hook in the ToolVerifier base class.
from .binding import DTIVerifier
from .expression import GeneExpressionVerifier

BIOLOGICAL_TOOLS = dict((x.name, x) for x in [GeneExpressionVerifier(), DTIVerifier()])
