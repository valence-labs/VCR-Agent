import re
from functools import cache
from collections.abc import Sequence
from collections import defaultdict
from typing import Any

import requests
from loguru import logger

_TRANSLOC_REGEXES: tuple[re.Pattern, ...] = (
    # e.g., "translocates from cytoplasm to nucleus"
    re.compile(
        r"translocat\w*\s+(?:from\s+)?(?P<from>[^.;:,()]+?)\s+(?:to|into|onto)\s+(?P<to>[^.;:,()]+)",
        re.IGNORECASE,
    ),
    # e.g., "cytoplasmic protein translocates to the plasma membrane"
    re.compile(
        r"(?P<from>cytoplasm(?:ic)?|nucleus|nucleolus|mitochondria|endoplasmic reticulum|golgi|plasma membrane|cell membrane|membrane|lysosome|peroxisome|endosome|cytosol)[^.;:,()]*?\s+translocat\w*\s+(?:to|into|onto)\s+(?P<to>[^.;:,()]+)",
        re.IGNORECASE,
    ),
)

ALIAS_GROUPS = [
    ["plasma membrane", "cell membrane", "cell surface", "plasmalemma", "sarcolemma", "pm"],
    ["cytoplasm", "cytosol", "cytoplasmic"],
    ["nucleus", "nuclear", "nucleoplasm"],
    ["nucleolus", "nucleolar"],
    ["golgi", "golgi apparatus", "golgi complex"],
    ["endoplasmic reticulum", "er"],
    ["mitochondria", "mitochondrion", "mito"],
    ["peroxisome", "microbody"],
    ["er-golgi intermediate compartment", "endoplasmic reticulum-golgi intermediate compartment", "ergic"],
    ["endosome", "endosomal"],
    ["lysosome", "lysosomal"],
    ["microtubule organizing center", "mtoc"],
    ["extracellular region", "extracellular space", "secreted", "outside cell"],
]

class LocalizationClient:
    """
    Client for subcellular localization analysis.

    Features:
      - Fetches subcellular locations from UniProt and/or Human Protein Atlas (HPA).
      - Checks membership and co-localization.
      - Extracts known/likely translocation events:
          * UniProt: regex-based extraction from "Subcellular location" note text.
          * HPA (optional): heuristic pairs 'main_location -> additional_location'.

    Args:
        sources: which sources to use for locations. Any of {"uniprot", "hpa"}.
        timeout_s: requests timeout (seconds).
    """

    def __init__(self, sources: Sequence[str] = ("uniprot", "hpa"), timeout_s: int = 15):
        self.sources = sources
        self.timeout_s = timeout_s

    @cache
    def get_locations(self, uniprot_ac: str) -> list[str]:
        """Get subcellular locations for a protein."""
        locations: list[str] = []

        srcs = set(s.lower() for s in self.sources)
        if "uniprot" in srcs:
            locations.extend(self._get_uniprot_locations(uniprot_ac))
        if "hpa" in srcs:
            locations.extend(self._get_hpa_locations(uniprot_ac))

        normed = {self._normalize_location(loc) for loc in locations if loc}
        normed.discard("")
        return sorted(normed)


    def is_located_in(self, uniprot_ac: str, location: str, known_locations: list[str] = None) -> bool:
        """Check if protein is located in a specific location."""
        locations = known_locations or self.get_locations(uniprot_ac)
        query = self._normalize_location(location)

        for loc in locations:
            if query.lower() in loc.lower() or loc.lower() in query.lower():
                return True
        return False

    def can_translocate(self, uniprot_ac: str, from_loc: str, to_loc: str, known_locations: list[str] = None) -> bool:
        """Heuristic: if protein is annotated in both compartments, it *can* translocate."""
        from_found = self.is_located_in(uniprot_ac, from_loc, known_locations)
        to_found = self.is_located_in(uniprot_ac, to_loc, known_locations)
        return from_found and to_found


    def get_translocations(self, uniprot_ac: str) -> list[dict[str, Any]]:
        """
        Heuristic: if protein is annotated in both compartments, it *can* translocate.
        Likely not worth using, just check can translocate between the two compartments that are proposed instead.
        """
        locations = self.get_locations(uniprot_ac)

        # --- alias helpers (your ALIAS_GROUPS is assumed global) ---
        def canonical(name: str) -> str:
            n = name.lower()
            for g in ALIAS_GROUPS:
                if n in g:
                    return g[0]  # stable canonical
            return n

        def same_alias(a: str, b: str) -> bool:
            return canonical(a) == canonical(b)

        # --- light bucket map (only to drop intra-family pairs) ---
        BUCKETS = {
            # nucleus family
            "nucleus": "nucleus", "nucleoplasm": "nucleus", "nuclear": "nucleus", "nucleolus": "nucleus",
            # cytoplasm family
            "cytoplasm": "cytoplasm", "cytosol": "cytoplasm", "cytoskeleton": "cytoplasm",
            # mtoc family
            "microtubule organizing center": "mtoc", "mtoc": "mtoc",
            "centrosome": "mtoc", "spindle pole body": "mtoc", "basal body": "mtoc",
            # membranes
            "plasma membrane": "plasma_membrane", "cell membrane": "plasma_membrane",
            "membrane": "membrane_generic",
            # organelles (kept distinct so pairs remain permissive)
            "golgi": "golgi", "golgi apparatus": "golgi", "golgi complex": "golgi",
            "endoplasmic reticulum": "er", "er": "er",
            "mitochondria": "mitochondria", "mitochondrion": "mitochondria",
            "lysosome": "lysosome", "endosome": "endosome", "peroxisome": "peroxisome",
            "extracellular region": "extracellular", "extracellular space": "extracellular",
        }
        def bucket_of(loc: str) -> str:
            return BUCKETS.get(loc.lower(), f"other::{loc.lower()}")

        def is_attribute(tok: str) -> bool:
            t = tok.lower()
            return "protein" in t or "topology" in t or "orientation" in t

        def split_tokens(s: str) -> list[str]:
            toks = [self._normalize_location(t) for t in s.split(",")]
            return [canonical(t) for t in toks if t and not is_attribute(t)]

        # --- parse into paths + singletons ---
        paths: list[list[str]] = []
        singles: set[str] = set()
        for entry in locations:
            if not entry:
                continue
            entry = entry.strip()
            if "," in entry:
                toks = split_tokens(entry)
                if len(toks) >= 2:
                    paths.append(toks)
                elif len(toks) == 1:
                    singles.add(toks[0])
            else:
                tok = canonical(self._normalize_location(entry))
                if tok and not is_attribute(tok):
                    singles.add(tok)

        # leaves + singletons
        leaves = {p[-1] for p in paths if p}
        primary = sorted({*leaves, *singles})
        # drop generic 'membrane' from heuristic pairs (keep specific membranes)
        primary_for_pairs = [p for p in primary if p != "membrane"]

        events: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        # --- (1) heuristic all-pairs (light pruning) ---
        for i, a in enumerate(primary_for_pairs):
            for j, b in enumerate(primary_for_pairs):
                if i == j or same_alias(a, b):
                    continue
                # prune only same-bucket pairs (e.g., mtoc↔mtoc, cytoplasm family internals)
                if bucket_of(a) == bucket_of(b):
                    continue
                key = (a, b, "heuristic")
                if key not in seen:
                    seen.add(key)
                    events.append(
                        {"from": a, "to": b,
                        "evidence": "HPA/UniProt leaves all-pairs (heuristic)",
                        "note": "Alias-normalized; intra-bucket pairs pruned"}
                    )

        # --- (2) path edges (always kept) ---
        for p in paths:
            for k in range(len(p) - 1):
                a, b = p[k], p[k + 1]
                if same_alias(a, b):
                    continue
                key = (a, b, "path")
                if key not in seen:
                    seen.add(key)
                    events.append(
                        {"from": a, "to": b,
                        "evidence": "UniProt/HPA path (hierarchy)",
                        "note": "Consecutive nodes from path-like location"}
                    )

        return events

    def _get_uniprot_locations(self, uniprot_ac: str) -> list[str]:
        """Get subcellular locations from UniProt (structured fields)."""
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_ac}?fields=cc_subcellular_location"
            resp = requests.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()

            locations: list[str] = []
            comments = data.get("comments", [])
            for c in comments:
                if c.get("commentType") != "SUBCELLULAR LOCATION":
                    continue
                for s in c.get("subcellularLocations", []) or []:
                    loc = ((s.get("location") or {}).get("value") or "").strip()
                    if loc:
                        locations.append(loc)
                    topo = ((s.get("topology") or {}).get("value") or "").strip()
                    if topo:
                        locations.append(topo)
                    orient = ((s.get("orientation") or {}).get("value") or "").strip()
                    if orient:
                        locations.append(orient)

            return locations

        except Exception as e:
            logger.warning(f"Failed to get UniProt locations for {uniprot_ac}: {e}")
            return []

    def _get_uniprot_subcellular_comments(self, uniprot_ac: str) -> list[dict[str, Any]]:
        """Return all UniProt 'SUBCELLULAR LOCATION' comments (full objects)."""
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_ac}?fields=cc_subcellular_location"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        comments = data.get("comments", [])
        return [c for c in comments if c.get("commentType") == "SUBCELLULAR LOCATION"]


    def _get_hpa_locations(self, uniprot_ac: str) -> list[str]:
        """Fetch HPA subcellular locations (flat schema) and return a deduped list."""
        all_locs = self._get_hpa_location_dict(uniprot_ac)
        return sorted(set([l for loc in all_locs.values() for l in loc]))

    def _get_hpa_location_dict(self, uniprot_ac: str) -> list[str]:
        """Fetch HPA subcellular locations (flat schema) and return a deduped list."""
        try:
            ensg_ids = self._uniprot_to_ensembl_ids(uniprot_ac)
            if not ensg_ids:
                return []

            def get_ci(d: dict, key: str, default=None):
                for k, v in d.items():
                    if k.lower() == key.lower():
                        return v
                return default

            all_locs: dict[str, list[str]] = defaultdict(list)
            for ensg in ensg_ids:
                try:
                    resp = requests.get(f"https://www.proteinatlas.org/{ensg}.json", timeout=self.timeout_s)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    hpa = resp.json()

                    main = get_ci(hpa, "Subcellular main location", []) or []
                    add  = get_ci(hpa, "Subcellular additional location", []) or []
                    any_ = get_ci(hpa, "Subcellular location", []) or []

                    # normalize to lists
                    
                    if isinstance(main, str): main = [main]
                    if isinstance(add, str):  add  = [add]
                    if isinstance(any_, str): any_ = [any_]

                    all_locs["main"].extend(main)
                    all_locs["additional"].extend(add)
                    all_locs["any"].extend(any_)

                except Exception as e:
                    logger.debug(f"HPA fetch/parse failed for {ensg}: {e}")
                    continue

            return all_locs

        except Exception as e:
            logger.warning(f"Failed to get HPA locations for {uniprot_ac}: {e}")
            return {}


    def _uniprot_to_ensembl_ids(self, uniprot_ac: str) -> list[str]:
        """Map UniProt accession to Ensembl Gene IDs via UniProt xrefs."""
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_ac}?fields=xref_ensembl"
            resp = requests.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            xrefs = data.get("uniProtKBCrossReferences", []) or []
            ensg: list[str] = []
            for x in xrefs:
                if (x.get("database") or "").lower() != "ensembl":
                    continue
                props = x.get("properties") or []
                for p in props:
                    if p.get("key") in ("GeneId", "gene ID", "Gene"):
                        val = (p.get("value") or "").strip()
                        if val.startswith("ENSG"):
                            ensg.append(val)
                xid = (x.get("id") or "").strip()
                if xid.startswith("ENSG"):
                    ensg.append(xid)
            return [x.split(".")[0] for x in sorted(set(ensg))]
        except Exception as e:
            logger.debug(f"Failed Ensembl mapping for {uniprot_ac}: {e}")
            return []

    def _extract_translocations_regex(self, note_text: str) -> list[tuple[str, str]]:
        """Regex-based 'from → to' extraction."""
        pairs: list[tuple[str, str]] = []
        for pat in self._TRANSLOC_REGEXES:
            for m in pat.finditer(note_text):
                frm = self._normalize_location(m.group("from"))
                to = self._normalize_location(m.group("to"))
                if frm and to and frm != to:
                    pairs.append((frm, to))
        # dedup
        seen = set()
        unique: list[tuple[str, str]] = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    @staticmethod
    def _normalize_location(loc: str) -> str:
        """Normalize a location string."""
        s = (loc or "").strip().strip(".,;:").lower()
        s = re.sub(r"\s+", " ", s)

        replacements = {
            "cell membrane": "plasma membrane",
            "pm": "plasma membrane",
            "cytoplasmic": "cytoplasm",
            "mitochondrion": "mitochondria",
            "golgi apparatus": "golgi",
            "endoplasmic reticulum membrane": "endoplasmic reticulum",
            "er": "endoplasmic reticulum",
            "nuclear": "nucleus",
        }
        return replacements.get(s, s)
