"""Knowledge graph service with comprehensive query methods."""

import re
from datetime import date
from typing import Any

from explain.eval.tools.knowledge_graph.client import GraphClient
from explain.eval.tools.knowledge_graph.models import Article, EntityFact, EntityRelationship


class KnowledgeGraphService:
    """High-level service for querying biomedical knowledge graph."""

    def __init__(self, client: GraphClient = None):
        self.client = client or GraphClient()

    def _normalize_entity_id(self, entity_id: str) -> str:
        """Normalize entity ID to internal format."""
        if entity_id.startswith("ENSEMBL:ENSG"):
            # Convert ENSEMBL:ENSG00000139618 to cbs:ensembl_ENSG00000139618
            ensg = entity_id.split(":")[1]
            return f"cbs:ensembl_{ensg}"
        elif entity_id.startswith("cbs:"):
            return entity_id
        elif entity_id.startswith("ENSG"):
            return f"cbs:ensembl_{entity_id}"
        else:
            return entity_id

    async def get_article_information(self, articles: list[str]) -> dict[str, Article]:
        """Get information about publications."""
        # Extract PMIDs from article references
        pmids = {}
        for article in articles:
            numbers = re.findall(r"\d+", article)
            if numbers:
                pmids[str(numbers[0])] = article

        if not pmids:
            return {}

        pmid_list = ",".join([f'"{pmid}"' for pmid in pmids.keys()])

        query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            SELECT ?id ?journal ?date (COUNT(DISTINCT ?other) AS ?citations) WHERE {{
                ?p a cbs:Publication ;
                   cbs:hasPubMedId ?id ;
                   cbs:publishedAt ?date .
                ?other cbs:cites ?p .
                OPTIONAL {{
                    ?p cbs:publishedIn/cbs:hasName ?journal .
                }}
                FILTER(?id IN ({pmid_list}))
            }}
            GROUP BY ?id ?journal ?date
        """

        try:
            results = await self.client.query(query)

            articles_info = {}
            for row in results:
                pmid = row["?id"]
                if pmid in pmids:
                    articles_info[pmids[pmid]] = Article(
                        citations=int(row.get("?citations", 0)),
                        journal=row.get("?journal", ""),
                        published=date.fromisoformat(row["?date"].split("T")[0]) if row.get("?date") else None,
                    )

            return articles_info

        except (RuntimeError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to get article information: {str(e)}") from e

    async def get_comprehensive_gene_information(self, gene_id: str) -> dict[str, Any]:
        """
        Get ALL available information about a gene in a single comprehensive query.
        This mega-query fetches everything needed for perturbation analysis and hypothesis testing.
        """
        gene_uri = self._normalize_entity_id(gene_id)

        # Single mega-query to get ALL gene information
        comprehensive_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            
            SELECT DISTINCT
                ?symbol ?fullName ?description ?geneType ?entrezId ?ensemblId
                ?synonym ?synonymType ?synonymSource
                ?altSymbol
                ?groupName ?groupDescription ?groupType ?groupCategory ?groupMember
                ?tissueName ?tissueType ?tissueDescription ?expressionLevel ?expressionSpecificity ?developmentStage
                ?bpName ?bpId ?bpDefinition ?bpCategory ?bpEvidenceCode
                ?mfName ?mfId ?mfDefinition ?mfCategory
                ?ccName ?ccId ?ccDefinition
                ?pathwayName ?pathwayId ?pathwayDescription ?pathwaySource
                ?proteinName ?proteinFamily ?proteinDomain ?proteinFunction
                ?regulatorGene ?regulatorSymbol ?regulationType
                ?targetGene ?targetSymbol ?targetRegulationType
                ?interactorGene ?interactorSymbol ?interactionType ?interactionScore
                ?chromosome ?startPos ?endPos ?strand
                ?compoundName ?compoundId ?activityType ?activityValue ?activityUnit ?assayType
                ?diseaseAssociation ?diseaseAssociationName ?associationType ?diseaseEvidence
                ?phenotypeName ?phenotypeId ?phenotypeDescription
            WHERE {{
                # Basic gene information
                {gene_uri} a ?geneType ;
                          cbs:hasApprovedSymbol ?symbol .
                OPTIONAL {{ {gene_uri} cbs:hasFullName ?fullName }}
                OPTIONAL {{ {gene_uri} rdfs:comment ?description }}
                OPTIONAL {{ {gene_uri} cbs:hasEntrezId ?entrezId }}
                OPTIONAL {{ {gene_uri} cbs:hasId ?ensemblId }}
                
                # All synonyms with source information
                OPTIONAL {{
                    {gene_uri} cbs:hasSynonym ?synonymEntity .
                    ?synonymEntity cbs:hasName ?synonym .
                    OPTIONAL {{ ?synonymEntity cbs:hasType ?synonymType }}
                    OPTIONAL {{ ?synonymEntity cbs:hasSource ?synonymSource }}
                }}
                
                # Alternative symbols
                OPTIONAL {{ {gene_uri} cbs:hasAlternativeSymbol ?altSymbol }}
                
                # Gene groups WITH comprehensive details AND other members
                OPTIONAL {{
                    {gene_uri} cbs:isMemberOf ?group .
                    ?group cbs:hasName ?groupName .
                    OPTIONAL {{ ?group cbs:hasDescription ?groupDescription }}
                    OPTIONAL {{ ?group cbs:hasType ?groupType }}
                    OPTIONAL {{ ?group cbs:hasCategory ?groupCategory }}
                    OPTIONAL {{
                        ?group cbs:hasMember ?otherMember .
                        ?otherMember cbs:hasApprovedSymbol ?groupMember .
                        FILTER(?otherMember != {gene_uri})
                    }}
                }}
                
                # Expression data WITH tissue context
                OPTIONAL {{
                    {gene_uri} cbs:expressedIn ?tissue .
                    ?tissue cbs:hasName ?tissueName .
                    OPTIONAL {{ ?tissue cbs:hasType ?tissueType }}
                    OPTIONAL {{ ?tissue cbs:hasDescription ?tissueDescription }}
                    OPTIONAL {{ {gene_uri} cbs:hasTpmExpressionLevel ?expressionLevel }}
                    OPTIONAL {{ {gene_uri} cbs:hasExpressionSpecificity ?expressionSpecificity }}
                    OPTIONAL {{ ?tissue cbs:hasDevelopmentStage ?developmentStage }}
                }}
                
                # Biological processes WITH definitions
                OPTIONAL {{
                    {gene_uri} cbs:associatesWithBiologicalProcess ?bp .
                    ?bp cbs:hasName ?bpName .
                    OPTIONAL {{ ?bp cbs:hasId ?bpId }}
                    OPTIONAL {{ ?bp cbs:hasDefinition ?bpDefinition }}
                    OPTIONAL {{ ?bp cbs:hasCategory ?bpCategory }}
                    OPTIONAL {{ ?bp cbs:hasEvidenceCode ?bpEvidenceCode }}
                }}
                
                # Molecular functions WITH definitions
                OPTIONAL {{
                    {gene_uri} cbs:encodesProtein ?protein .
                    ?protein cbs:hasMolecularFunction ?mf .
                    ?mf cbs:hasName ?mfName .
                    OPTIONAL {{ ?mf cbs:hasId ?mfId }}
                    OPTIONAL {{ ?mf cbs:hasDefinition ?mfDefinition }}
                    OPTIONAL {{ ?mf cbs:hasCategory ?mfCategory }}
                }}
                
                # Cellular components
                OPTIONAL {{
                    {gene_uri} cbs:encodesProtein ?protein .
                    ?protein cbs:hasLocation ?cc .
                    ?cc cbs:hasName ?ccName .
                    OPTIONAL {{ ?cc cbs:hasId ?ccId }}
                    OPTIONAL {{ ?cc cbs:hasDefinition ?ccDefinition }}
                }}
                
                # Pathways WITH descriptions
                OPTIONAL {{
                    {gene_uri} cbs:participatesIn ?pathway .
                    ?pathway cbs:hasName ?pathwayName .
                    OPTIONAL {{ ?pathway cbs:hasId ?pathwayId }}
                    OPTIONAL {{ ?pathway cbs:hasDescription ?pathwayDescription }}
                    OPTIONAL {{ ?pathway cbs:hasSource ?pathwaySource }}
                }}
                
                # Protein information WITH families and domains
                OPTIONAL {{
                    {gene_uri} cbs:encodesProtein ?protein .
                    OPTIONAL {{ ?protein cbs:hasName ?proteinName }}
                    OPTIONAL {{
                        ?protein cbs:belongsToFamily ?family .
                        ?family cbs:hasName ?proteinFamily
                    }}
                    OPTIONAL {{
                        ?protein cbs:containsDomain ?domain .
                        ?domain cbs:hasName ?proteinDomain
                    }}
                    OPTIONAL {{ ?protein cbs:hasFunction ?proteinFunction }}
                }}
                
                # Regulatory relationships - what regulates this gene
                OPTIONAL {{
                    ?regulator cbs:regulates {gene_uri} .
                    ?regulator cbs:hasApprovedSymbol ?regulatorSymbol .
                    BIND(?regulator AS ?regulatorGene)
                    BIND("regulated_by" AS ?regulationType)
                }}
                
                # Regulatory relationships - what this gene regulates
                OPTIONAL {{
                    {gene_uri} cbs:regulates ?target .
                    ?target cbs:hasApprovedSymbol ?targetSymbol .
                    BIND(?target AS ?targetGene)
                    BIND("regulates" AS ?targetRegulationType)
                }}
                
                # Protein-protein interactions WITH details
                OPTIONAL {{
                    {gene_uri} cbs:encodesProtein ?protein1 .
                    ?interaction cbs:hasParticipant ?protein1 ;
                                 cbs:hasParticipant ?protein2 .
                    ?interactorGene cbs:encodesProtein ?protein2 ;
                                   cbs:hasApprovedSymbol ?interactorSymbol .
                    OPTIONAL {{ ?interaction cbs:hasType ?interactionType }}
                    OPTIONAL {{ ?interaction cbs:hasInteractionScore ?interactionScore }}
                    FILTER(?protein1 != ?protein2)
                }}
                
                # Genomic location
                OPTIONAL {{
                    {gene_uri} cbs:hasChromosome ?chromosome ;
                              cbs:hasStart ?startPos ;
                              cbs:hasEnd ?endPos .
                    OPTIONAL {{ {gene_uri} cbs:hasStrand ?strand }}
                }}
                
                # Compounds that target this gene WITH activity data
                OPTIONAL {{
                    ?compound cbs:hasActivity ?activity .
                    ?activity cbs:targets {gene_uri} .
                    ?compound cbs:hasName ?compoundName .
                    OPTIONAL {{ ?compound cbs:hasId ?compoundId }}
                    OPTIONAL {{
                        ?activity cbs:hasType ?activityType ;
                                 cbs:hasMeasurement ?measurement .
                        ?measurement cbs:hasValue ?activityValue .
                        OPTIONAL {{ ?measurement cbs:hasUnit ?activityUnit }}
                    }}
                    OPTIONAL {{
                        ?activity cbs:byAssay ?assay .
                        ?assay cbs:hasType ?assayType
                    }}
                }}
                
                # Disease associations WITH evidence
                OPTIONAL {{
                    {{
                        {gene_uri} cbs:associatesWithDisease ?diseaseAssociation .
                        BIND("associates_with" AS ?associationType)
                    }}
                    UNION
                    {{
                        {gene_uri} cbs:causesDisease ?diseaseAssociation .
                        BIND("causes" AS ?associationType)
                    }}
                    ?diseaseAssociation cbs:hasName ?diseaseAssociationName .
                    OPTIONAL {{ ?diseaseAssociation cbs:hasEvidence ?diseaseEvidence }}
                }}
                
                # Phenotype associations
                OPTIONAL {{
                    {gene_uri} cbs:associatesWithPhenotype ?phenotype .
                    ?phenotype cbs:hasName ?phenotypeName .
                    OPTIONAL {{ ?phenotype cbs:hasId ?phenotypeId }}
                    OPTIONAL {{ ?phenotype cbs:hasDescription ?phenotypeDescription }}
                }}
                
                FILTER(?geneType != <http://www.w3.org/2002/07/owl#NamedIndividual>)
            }}
        """

        try:
            results = await self.client.query(comprehensive_query)

            # Process and organize the comprehensive results
            gene_info = {
                "gene_id": gene_id,
                "basic_info": {},
                "synonyms": [],
                "gene_groups": [],
                "expression_profile": [],
                "functional_annotations": {
                    "biological_processes": [],
                    "molecular_functions": [],
                    "cellular_components": [],
                    "pathways": [],
                },
                "protein_information": {},
                "regulatory_network": {"regulators": [], "targets": []},
                "interactions": [],
                "genomic_location": {},
                "targeting_compounds": [],
                "disease_associations": [],
                "phenotype_associations": [],
            }

            # Track groups to aggregate members
            groups_dict = {}

            # Process results and populate structured data
            for result in results:
                # Basic gene info (only set once)
                if "?symbol" in result and not gene_info["basic_info"]:
                    gene_info["basic_info"] = {
                        "symbol": result.get("?symbol", ""),
                        "full_name": result.get("?fullName", ""),
                        "description": result.get("?description", ""),
                        "gene_type": result.get("?geneType", "").split("#")[-1] if result.get("?geneType") else "",
                        "entrez_id": result.get("?entrezId", ""),
                        "ensembl_id": result.get("?ensemblId", ""),
                    }

                # Synonyms (with deduplication)
                if "?synonym" in result and result["?synonym"]:
                    synonym_info = {
                        "name": result["?synonym"],
                        "type": result.get("?synonymType", ""),
                        "source": result.get("?synonymSource", ""),
                    }
                    if synonym_info not in gene_info["synonyms"]:
                        gene_info["synonyms"].append(synonym_info)

                # Gene groups WITH members aggregation
                if "?groupName" in result and result["?groupName"]:
                    group_name = result["?groupName"]
                    if group_name not in groups_dict:
                        groups_dict[group_name] = {
                            "name": group_name,
                            "description": result.get("?groupDescription", ""),
                            "type": result.get("?groupType", ""),
                            "category": result.get("?groupCategory", ""),
                            "other_members": set(),
                        }

                    # Add other group members
                    if "?groupMember" in result and result["?groupMember"]:
                        groups_dict[group_name]["other_members"].add(result["?groupMember"])

                # Expression profile
                if "?tissueName" in result and result["?tissueName"]:
                    expression_info = {
                        "tissue_name": result["?tissueName"],
                        "tissue_type": result.get("?tissueType", ""),
                        "tissue_description": result.get("?tissueDescription", ""),
                        "expression_level": result.get("?expressionLevel", ""),
                        "expression_specificity": result.get("?expressionSpecificity", ""),
                        "development_stage": result.get("?developmentStage", ""),
                    }
                    if expression_info not in gene_info["expression_profile"]:
                        gene_info["expression_profile"].append(expression_info)

                # Biological processes
                if "?bpName" in result and result["?bpName"]:
                    bp_info = {
                        "name": result["?bpName"],
                        "id": result.get("?bpId", ""),
                        "definition": result.get("?bpDefinition", ""),
                        "category": result.get("?bpCategory", ""),
                        "evidence_code": result.get("?bpEvidenceCode", ""),
                    }
                    if bp_info not in gene_info["functional_annotations"]["biological_processes"]:
                        gene_info["functional_annotations"]["biological_processes"].append(bp_info)

                # Molecular functions
                if "?mfName" in result and result["?mfName"]:
                    mf_info = {
                        "name": result["?mfName"],
                        "id": result.get("?mfId", ""),
                        "definition": result.get("?mfDefinition", ""),
                        "category": result.get("?mfCategory", ""),
                    }
                    if mf_info not in gene_info["functional_annotations"]["molecular_functions"]:
                        gene_info["functional_annotations"]["molecular_functions"].append(mf_info)

                # Cellular components
                if "?ccName" in result and result["?ccName"]:
                    cc_info = {
                        "name": result["?ccName"],
                        "id": result.get("?ccId", ""),
                        "definition": result.get("?ccDefinition", ""),
                    }
                    if cc_info not in gene_info["functional_annotations"]["cellular_components"]:
                        gene_info["functional_annotations"]["cellular_components"].append(cc_info)

                # Pathways
                if "?pathwayName" in result and result["?pathwayName"]:
                    pathway_info = {
                        "name": result["?pathwayName"],
                        "id": result.get("?pathwayId", ""),
                        "description": result.get("?pathwayDescription", ""),
                        "source": result.get("?pathwaySource", ""),
                    }
                    if pathway_info not in gene_info["functional_annotations"]["pathways"]:
                        gene_info["functional_annotations"]["pathways"].append(pathway_info)

                # Protein interactions
                if "?interactorSymbol" in result and result["?interactorSymbol"]:
                    interaction_info = {
                        "interactor_gene": result.get("?interactorGene", ""),
                        "interactor_symbol": result["?interactorSymbol"],
                        "interaction_type": result.get("?interactionType", ""),
                        "interaction_score": result.get("?interactionScore", ""),
                    }
                    if interaction_info not in gene_info["interactions"]:
                        gene_info["interactions"].append(interaction_info)

                # Regulatory relationships
                if "?regulatorSymbol" in result and result["?regulatorSymbol"]:
                    regulator_info = {
                        "regulator_gene": result.get("?regulatorGene", ""),
                        "regulator_symbol": result["?regulatorSymbol"],
                        "regulation_type": result.get("?regulationType", ""),
                    }
                    if regulator_info not in gene_info["regulatory_network"]["regulators"]:
                        gene_info["regulatory_network"]["regulators"].append(regulator_info)

                if "?targetSymbol" in result and result["?targetSymbol"]:
                    target_info = {
                        "target_gene": result.get("?targetGene", ""),
                        "target_symbol": result["?targetSymbol"],
                        "regulation_type": result.get("?targetRegulationType", ""),
                    }
                    if target_info not in gene_info["regulatory_network"]["targets"]:
                        gene_info["regulatory_network"]["targets"].append(target_info)

                # Targeting compounds
                if "?compoundName" in result and result["?compoundName"]:
                    compound_info = {
                        "compound_name": result["?compoundName"],
                        "compound_id": result.get("?compoundId", ""),
                        "activity_type": result.get("?activityType", ""),
                        "activity_value": result.get("?activityValue", ""),
                        "activity_unit": result.get("?activityUnit", ""),
                        "assay_type": result.get("?assayType", ""),
                    }
                    if compound_info not in gene_info["targeting_compounds"]:
                        gene_info["targeting_compounds"].append(compound_info)

                # Disease associations
                if "?diseaseAssociationName" in result and result["?diseaseAssociationName"]:
                    disease_info = {
                        "disease_name": result["?diseaseAssociationName"],
                        "association_type": result.get("?associationType", ""),
                        "evidence": result.get("?diseaseEvidence", ""),
                    }
                    if disease_info not in gene_info["disease_associations"]:
                        gene_info["disease_associations"].append(disease_info)

                # Phenotype associations
                if "?phenotypeName" in result and result["?phenotypeName"]:
                    phenotype_info = {
                        "phenotype_name": result["?phenotypeName"],
                        "phenotype_id": result.get("?phenotypeId", ""),
                        "description": result.get("?phenotypeDescription", ""),
                    }
                    if phenotype_info not in gene_info["phenotype_associations"]:
                        gene_info["phenotype_associations"].append(phenotype_info)

                # Genomic location (set once)
                if "?chromosome" in result and result["?chromosome"] and not gene_info["genomic_location"]:
                    gene_info["genomic_location"] = {
                        "chromosome": result["?chromosome"],
                        "start": result.get("?startPos", ""),
                        "end": result.get("?endPos", ""),
                        "strand": result.get("?strand", ""),
                    }

            # Convert gene groups with aggregated members
            for group_data in groups_dict.values():
                group_data["other_members"] = sorted(list(group_data["other_members"]))
                gene_info["gene_groups"].append(group_data)

            return gene_info

        except Exception as e:
            raise RuntimeError(f"Failed to get comprehensive gene information: {str(e)}") from e

    async def get_comprehensive_perturbation_effects(
        self, perturbation_id: str, perturbation_type: str, cellular_context: str = None
    ) -> dict[str, Any]:
        """
        Get ALL perturbation effects and associated information in a single comprehensive query.
        Covers compounds, gene perturbations, and their effects across different cellular contexts.
        """
        perturbation_uri = self._normalize_entity_id(perturbation_id)

        # Mega-query for comprehensive perturbation analysis
        perturbation_query = f"""
            PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT
                ?perturbationName ?perturbationDescription ?perturbationType
                ?targetGene ?targetSymbol ?targetName ?targetType
                ?activityId ?activityType ?activityDescription
                ?measurementValue ?measurementUnit ?measurementType ?measurementMethod
                ?assayId ?assayName ?assayType ?assayDescription ?assayProtocol
                ?cellLineName ?cellLineType ?cellLineDescription ?cellLineOrigin
                ?tissueContext ?organismContext
                ?doseValue ?doseUnit ?timePoint ?concentration
                ?phenotypeEffect ?phenotypeDescription ?phenotypeSeverity
                ?pathwayAffected ?pathwayDescription ?pathwayType
                ?mechanismType ?mechanismDescription
                ?sideEffect ?sideEffectSeverity ?sideEffectFrequency
                ?selectivityTarget ?selectivityRatio ?offTargetEffect
                ?efficacyMeasure ?potencyValue ?therapeuticWindow
                ?metaboliteId ?metaboliteName ?metabolismPathway
                ?resistanceMechanism ?resistanceFrequency
                ?combinationEffect ?combinationSynergy ?combinationDrug
            WHERE {{
                # Basic perturbation information
                {perturbation_uri} cbs:hasName ?perturbationName .
                OPTIONAL {{ {perturbation_uri} rdfs:comment ?perturbationDescription }}
                OPTIONAL {{ {perturbation_uri} a ?perturbationType }}
                
                # Direct target information WITH comprehensive details
                OPTIONAL {{
                    {perturbation_uri} cbs:hasActivity ?activityId .
                    ?activityId cbs:targets ?targetGene .
                    ?targetGene cbs:hasApprovedSymbol ?targetSymbol .
                    OPTIONAL {{ ?targetGene cbs:hasName ?targetName }}
                    OPTIONAL {{ ?targetGene a ?targetType }}
                    
                    # Activity details
                    OPTIONAL {{ ?activityId cbs:hasType ?activityType }}
                    OPTIONAL {{ ?activityId cbs:hasDescription ?activityDescription }}
                    
                    # Measurement data WITH full context
                    OPTIONAL {{
                        ?activityId cbs:hasMeasurement ?measurement .
                        ?measurement cbs:hasValue ?measurementValue .
                        OPTIONAL {{ ?measurement cbs:hasUnit ?measurementUnit }}
                        OPTIONAL {{ ?measurement cbs:hasType ?measurementType }}
                        OPTIONAL {{ ?measurement cbs:hasMethod ?measurementMethod }}
                    }}
                    
                    # Assay information WITH protocols
                    OPTIONAL {{
                        ?activityId cbs:byAssay ?assayId .
                        ?assayId cbs:hasName ?assayName .
                        OPTIONAL {{ ?assayId cbs:hasType ?assayType }}
                        OPTIONAL {{ ?assayId cbs:hasDescription ?assayDescription }}
                        OPTIONAL {{ ?assayId cbs:hasProtocol ?assayProtocol }}
                    }}
                    
                    # Cellular context WITH comprehensive details
                    OPTIONAL {{
                        ?activityId cbs:hasContext ?cellContext .
                        ?cellContext cbs:hasName ?cellLineName .
                        OPTIONAL {{ ?cellContext cbs:hasType ?cellLineType }}
                        OPTIONAL {{ ?cellContext cbs:hasDescription ?cellLineDescription }}
                        OPTIONAL {{ ?cellContext cbs:hasOrigin ?cellLineOrigin }}
                        OPTIONAL {{ ?cellContext cbs:derivedFromTissue/cbs:hasName ?tissueContext }}
                        OPTIONAL {{ ?cellContext cbs:hasOrganism/cbs:hasName ?organismContext }}
                    }}
                    
                    # Dose-response information
                    OPTIONAL {{ ?activityId cbs:hasDose ?doseValue }}
                    OPTIONAL {{ ?activityId cbs:hasDoseUnit ?doseUnit }}
                    OPTIONAL {{ ?activityId cbs:hasTimePoint ?timePoint }}
                    OPTIONAL {{ ?activityId cbs:hasConcentration ?concentration }}
                }}
                
                # Phenotypic effects WITH severity
                OPTIONAL {{
                    {perturbation_uri} cbs:causes ?phenotypeEffect .
                    OPTIONAL {{ ?phenotypeEffect cbs:hasDescription ?phenotypeDescription }}
                    OPTIONAL {{ ?phenotypeEffect cbs:hasSeverity ?phenotypeSeverity }}
                }}
                
                # Pathway effects
                OPTIONAL {{
                    {perturbation_uri} cbs:affects ?pathwayAffected .
                    ?pathwayAffected cbs:hasName ?pathwayAffected .
                    OPTIONAL {{ ?pathwayAffected cbs:hasDescription ?pathwayDescription }}
                    OPTIONAL {{ ?pathwayAffected cbs:hasType ?pathwayType }}
                }}
                
                # Mechanism of action
                OPTIONAL {{
                    {perturbation_uri} cbs:hasMechanism ?mechanism .
                    ?mechanism cbs:hasType ?mechanismType .
                    OPTIONAL {{ ?mechanism cbs:hasDescription ?mechanismDescription }}
                }}
                
                # Side effects WITH frequency data
                OPTIONAL {{
                    {perturbation_uri} cbs:hasSideEffect ?sideEffect .
                    OPTIONAL {{ ?sideEffect cbs:hasSeverity ?sideEffectSeverity }}
                    OPTIONAL {{ ?sideEffect cbs:hasFrequency ?sideEffectFrequency }}
                }}
                
                # Selectivity and off-target effects
                OPTIONAL {{
                    {perturbation_uri} cbs:hasSelectivity ?selectivity .
                    ?selectivity cbs:hasTarget ?selectivityTarget .
                    OPTIONAL {{ ?selectivity cbs:hasRatio ?selectivityRatio }}
                    OPTIONAL {{ ?selectivity cbs:hasOffTargetEffect ?offTargetEffect }}
                }}
                
                # Efficacy and potency data
                OPTIONAL {{ {perturbation_uri} cbs:hasEfficacy ?efficacyMeasure }}
                OPTIONAL {{ {perturbation_uri} cbs:hasPotency ?potencyValue }}
                OPTIONAL {{ {perturbation_uri} cbs:hasTherapeuticWindow ?therapeuticWindow }}
                
                # Metabolism information
                OPTIONAL {{
                    {perturbation_uri} cbs:hasMetabolite ?metaboliteId .
                    ?metaboliteId cbs:hasName ?metaboliteName .
                    OPTIONAL {{ ?metaboliteId cbs:producedBy ?metabolismPathway }}
                }}
                
                # Resistance mechanisms
                OPTIONAL {{
                    {perturbation_uri} cbs:hasResistance ?resistance .
                    ?resistance cbs:hasMechanism ?resistanceMechanism .
                    OPTIONAL {{ ?resistance cbs:hasFrequency ?resistanceFrequency }}
                }}
                
                # Combination effects
                OPTIONAL {{
                    {perturbation_uri} cbs:hasCombinationWith ?combinationDrug .
                    ?combinationDrug cbs:hasName ?combinationDrug .
                    OPTIONAL {{ ?combinationDrug cbs:hasEffect ?combinationEffect }}
                    OPTIONAL {{ ?combinationDrug cbs:hasSynergy ?combinationSynergy }}
                }}
                
                # Filter cellular context if specified
                {f'FILTER(CONTAINS(LCASE(?cellLineName), LCASE("{cellular_context}")))' if cellular_context else ""}
            }}
        """

        try:
            results = await self.client.query(perturbation_query)

            # Organize comprehensive perturbation data
            perturbation_info = {
                "perturbation_id": perturbation_id,
                "perturbation_type": perturbation_type,
                "basic_info": {},
                "direct_targets": [],
                "activity_profiles": [],
                "cellular_contexts": [],
                "phenotypic_effects": [],
                "pathway_effects": [],
                "mechanisms": [],
                "side_effects": [],
                "selectivity_profile": [],
                "pharmacological_properties": {},
                "metabolism": [],
                "resistance_mechanisms": [],
                "combination_effects": [],
            }

            # Process all results with comprehensive data aggregation
            for result in results:
                # Basic perturbation info (set once)
                if "?perturbationName" in result and not perturbation_info["basic_info"]:
                    perturbation_info["basic_info"] = {
                        "name": result.get("?perturbationName", ""),
                        "description": result.get("?perturbationDescription", ""),
                        "type": result.get("?perturbationType", "").split("#")[-1]
                        if result.get("?perturbationType")
                        else "",
                    }

                # Direct targets
                if "?targetSymbol" in result and result["?targetSymbol"]:
                    target_info = {
                        "target_gene": result.get("?targetGene", ""),
                        "target_symbol": result["?targetSymbol"],
                        "target_name": result.get("?targetName", ""),
                        "target_type": result.get("?targetType", "").split("#")[-1]
                        if result.get("?targetType")
                        else "",
                    }
                    if target_info not in perturbation_info["direct_targets"]:
                        perturbation_info["direct_targets"].append(target_info)

                # Activity profiles
                if "?activityId" in result and result["?activityId"]:
                    activity_info = {
                        "activity_id": result["?activityId"],
                        "activity_type": result.get("?activityType", ""),
                        "activity_description": result.get("?activityDescription", ""),
                        "measurement_value": result.get("?measurementValue", ""),
                        "measurement_unit": result.get("?measurementUnit", ""),
                        "measurement_type": result.get("?measurementType", ""),
                        "measurement_method": result.get("?measurementMethod", ""),
                        "dose_value": result.get("?doseValue", ""),
                        "dose_unit": result.get("?doseUnit", ""),
                        "time_point": result.get("?timePoint", ""),
                        "concentration": result.get("?concentration", ""),
                    }
                    if activity_info not in perturbation_info["activity_profiles"]:
                        perturbation_info["activity_profiles"].append(activity_info)

                # Set pharmacological properties (once)
                if not perturbation_info["pharmacological_properties"]:
                    perturbation_info["pharmacological_properties"] = {
                        "efficacy": result.get("?efficacyMeasure", ""),
                        "potency": result.get("?potencyValue", ""),
                        "therapeutic_window": result.get("?therapeuticWindow", ""),
                    }

            return perturbation_info

        except Exception as e:
            raise RuntimeError(f"Failed to get comprehensive perturbation effects: {str(e)}") from e

    # Keep legacy methods for backward compatibility but delegate to comprehensive versions
    async def get_gene_facts(self, gene_id: str, label: str = None) -> tuple[str, list[EntityFact]] | None:
        """Legacy method - use get_comprehensive_gene_information for better results."""
        # Note: label parameter is kept for backward compatibility but unused
        _ = label  # Acknowledge unused parameter
        try:
            comprehensive_data = await self.get_comprehensive_gene_information(gene_id)
            if not comprehensive_data.get("basic_info"):
                return None

            # Convert comprehensive data to legacy format
            facts = []
            basic_info = comprehensive_data["basic_info"]

            if basic_info.get("symbol"):
                facts.append(EntityFact(relationship="approved_symbol", value=basic_info["symbol"]))
            if basic_info.get("description"):
                facts.append(EntityFact(relationship="description", value=basic_info["description"]))

            # Add other facts from comprehensive data
            for synonym in comprehensive_data.get("synonyms", []):
                facts.append(EntityFact(relationship="synonym", value=synonym["name"]))

            return basic_info.get("symbol", gene_id), facts

        except Exception as e:
            raise RuntimeError(f"Failed to get gene facts: {str(e)}") from e

    async def find_relationships(
        self, source_id: str, source_type: str, target_type: str = None
    ) -> list[EntityRelationship]:
        """Find relationships between entities with enhanced context."""
        source_uri = self._normalize_entity_id(source_id)

        if source_type == "GENE" and target_type == "DISEASE":
            # Enhanced gene-disease associations
            query = f"""
                PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                SELECT DISTINCT ?target_entity ?target_name ?relationship_type ?evidence ?additional_info WHERE {{
                    {{
                        {source_uri} cbs:associatesWithDisease ?target_entity .
                        ?target_entity cbs:hasName ?target_name .
                        BIND("directly_associates_with_disease" AS ?relationship_type)
                        OPTIONAL {{ ?target_entity cbs:hasEvidence ?evidence }}
                    }}
                    UNION
                    {{
                        {source_uri} cbs:causesDisease ?target_entity .
                        ?target_entity cbs:hasName ?target_name .
                        BIND("causes_disease" AS ?relationship_type)
                    }}
                    UNION
                    {{
                        ?compound cbs:hasActivity ?activity .
                        ?activity cbs:targets {source_uri} .
                        ?compound cbs:associatesWithDisease ?target_entity .
                        ?target_entity cbs:hasName ?target_name .
                        BIND("disease_via_drug_interaction" AS ?relationship_type)
                        BIND(?compound AS ?additional_info)
                    }}
                }}
                LIMIT 50
            """
        elif source_type == "GENE" and target_type == "GENE":
            # Enhanced gene-gene interactions
            query = f"""
                PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
                SELECT DISTINCT ?target_entity ?target_name ?relationship_type ?additional_info WHERE {{
                    {{
                        {source_uri} cbs:encodesProtein ?proteinA .
                        ?interaction cbs:hasParticipant ?proteinA ;
                                     cbs:hasParticipant ?proteinB .
                        ?target_entity cbs:encodesProtein ?proteinB ;
                                      cbs:hasApprovedSymbol ?target_name .
                        OPTIONAL {{ ?interaction cbs:hasInteractionScore ?additional_info }}
                        BIND("protein_interaction" AS ?relationship_type)
                        FILTER(?proteinA != ?proteinB)
                    }}
                    UNION
                    {{
                        {source_uri} cbs:participatesIn ?pathway .
                        ?target_entity cbs:participatesIn ?pathway ;
                                      cbs:hasApprovedSymbol ?target_name .
                        ?pathway cbs:hasName ?additional_info .
                        BIND("shared_pathway" AS ?relationship_type)
                        FILTER({source_uri} != ?target_entity)
                    }}
                }}
                LIMIT 50
            """
        else:
            # Generic relationship query with better name resolution
            query = f"""
                PREFIX cbs: <https://www.exscientia.ai/cb_schema#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?predicate ?target_entity ?target_name WHERE {{
                    {source_uri} ?predicate ?target_entity .
                    OPTIONAL {{
                        {{
                            ?target_entity cbs:hasName ?target_name .
                        }}
                        UNION
                        {{
                            ?target_entity cbs:hasApprovedSymbol ?target_name .
                        }}
                        UNION
                        {{
                            ?target_entity rdfs:label ?target_name .
                        }}
                    }}
                    FILTER(isURI(?target_entity))
                    FILTER(?predicate != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
                }}
                LIMIT 50
            """

        try:
            results = await self.client.query(query)
            relationships = []

            for result in results:
                target_name = result.get("target_name", "")
                target_entity = result.get("target_entity", "")
                relationship_type = result.get("relationship_type", "generic_relation")

                # Extract additional info
                additional_info = {}
                if "evidence" in result and result["evidence"]:
                    additional_info["evidence"] = result["evidence"]
                if "additional_info" in result and result["additional_info"]:
                    additional_info["details"] = result["additional_info"]
                if "predicate" in result and result["predicate"]:
                    predicate_name = (
                        result["predicate"].split("#")[-1] if "#" in result["predicate"] else result["predicate"]
                    )
                    additional_info["predicate"] = predicate_name
                    if not relationship_type or relationship_type == "generic_relation":
                        relationship_type = predicate_name.replace("_", " ")

                # Generate descriptive information
                relationship_description = self._get_relationship_description(relationship_type)
                target_description = f"A {target_type.lower() if target_type else 'biological entity'}"
                evidence_level = additional_info.get("evidence", "Unknown")

                relationships.append(
                    EntityRelationship(
                        target_entity=target_entity,
                        target_name=target_name if target_name else "Unknown",
                        relationship_type=relationship_type,
                        target_description=target_description,
                        relationship_description=relationship_description,
                        evidence_level=evidence_level,
                        source_database="Knowledge Graph",
                        additional_info=additional_info,
                    )
                )

            return relationships

        except Exception as e:
            raise RuntimeError(f"Failed to find relationships: {str(e)}") from e

    def _get_relationship_description(self, relationship_type: str) -> str:
        """Generate descriptive text for relationship types."""
        descriptions = {
            "directly_associates_with_disease": "Gene has been directly linked to this disease through experimental evidence",
            "causes_disease": "Gene mutations or dysfunction directly cause this disease",
            "disease_via_drug_interaction": "Gene is linked to disease through drug interaction pathways",
            "protein_interaction": "Encoded proteins physically interact or form complexes",
            "shared_pathway": "Genes participate in the same biological pathway or process",
            "regulates": "Gene product regulates the expression or activity of target gene",
            "co_expressed_in": "Genes show coordinated expression patterns in the same tissues",
        }
        return descriptions.get(
            relationship_type, f"The entities have a {relationship_type.replace('_', ' ')} relationship"
        )
