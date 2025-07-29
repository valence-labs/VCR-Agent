"""
Response parser for extracting structured sections from LLM responses with entity awareness
"""

import re
from typing import Dict, Optional, List, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParsedResponse:
    """Structured response from LLM with entity information"""
    thinking: Optional[str] = None
    answer: Optional[str] = None
    explain: Optional[str] = None
    dag: Optional[str] = None
    raw_response: str = ""
    entity_mentions: Dict[str, List[str]] = None
    action_primitives_used: List[str] = None
    validation_issues: List[str] = None
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "thinking": self.thinking,
            "answer": self.answer, 
            "explain": self.explain,
            "dag": self.dag,
            "raw_response": self.raw_response,
            "entity_mentions": self.entity_mentions,
            "action_primitives_used": self.action_primitives_used,
            "validation_issues": self.validation_issues
        }

class ResponseParser:
    """Enhanced parser for extracting structured sections from LLM responses"""
    
    def __init__(self):
        self.patterns = {
            'thinking': re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL | re.IGNORECASE),
            'answer': re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL | re.IGNORECASE),
            'explain': re.compile(r'<explain>\s*(.*?)\s*</explain>', re.DOTALL | re.IGNORECASE),
            'dag': re.compile(r'<dag>\s*(.*?)\s*</dag>', re.DOTALL | re.IGNORECASE)
        }
        
        # Entity mention patterns
        self.entity_patterns = {
            'markdown': re.compile(r'\*\*([^*]+)\*\*\[([A-Z]+):([^\]]+)\]'),
            'html': re.compile(r'<abbr[^>]*title="([^"]*)"[^>]*>([^<]+)</abbr>'),
            'brackets': re.compile(r'([^{]+)\{([A-Z]+):([^}]+)\}'),
            'inline': re.compile(r'\[([A-Z]+):([^:]+):([^\]]+)\]'),
            'simple_bold': re.compile(r'\*\*([A-Z][A-Z0-9-]+)\*\*')  # Simple bold entities
        }
        
        # Common biomedical action patterns for validation
        self.biomedical_action_patterns = {
            'binding': re.compile(r'\b(bind[s]?|binding|bound)\b', re.IGNORECASE),
            'inhibition': re.compile(r'\b(inhibit[s]?|inhibition|inhibitor|block[s]?|blocking)\b', re.IGNORECASE),
            'activation': re.compile(r'\b(activate[s]?|activation|stimulate[s]?|enhance[s]?)\b', re.IGNORECASE),
            'expression': re.compile(r'\b(express[es]?|expression|upregulat[es]?|downregulat[es]?)\b', re.IGNORECASE),
            'pathway': re.compile(r'\b(pathway|signaling|cascade|network)\b', re.IGNORECASE)
        }
        
        # DAG edge pattern for validation
        self.dag_edge_pattern = re.compile(r'edge\s*\(\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*relation\s*=\s*["\']([^"\']+)["\']\s*\)', re.IGNORECASE)
    
    def parse_response(self, response: str) -> ParsedResponse:
        """Parse LLM response with entity awareness"""
        parsed = ParsedResponse(raw_response=response)
        
        # Extract main sections
        for section_name, pattern in self.patterns.items():
            match = pattern.search(response)
            if match:
                content = match.group(1).strip()
                setattr(parsed, section_name, content)
        
        # Extract entity mentions
        parsed.entity_mentions = self.extract_entity_mentions(response)
        
        # Extract action primitives used
        if parsed.explain:
            parsed.action_primitives_used = self.extract_action_primitives(parsed.explain)
        
        return parsed
    
    def extract_entity_mentions(self, text: str) -> Dict[str, List[str]]:
        """Extract entity mentions from response text"""
        entity_mentions = {}
        
        for pattern_name, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            
            if pattern_name == 'markdown':
                # Format: **VEGF**[GENE:ENSEMBL:ENSG00000112715]
                for entity_text, entity_type, entity_id in matches:
                    if entity_type not in entity_mentions:
                        entity_mentions[entity_type] = []
                    entity_mentions[entity_type].append(f"{entity_text} ({entity_id})")
            
            elif pattern_name == 'html':
                # Format: <abbr title="GENE:ENSEMBL:ENSG00000112715">VEGF</abbr>
                for title, entity_text in matches:
                    if ':' in title:
                        entity_type = title.split(':')[0]
                        entity_id = title.split(':')[1] if len(title.split(':')) > 1 else ''
                        if entity_type not in entity_mentions:
                            entity_mentions[entity_type] = []
                        entity_mentions[entity_type].append(f"{entity_text} ({entity_id})")
            
            elif pattern_name == 'brackets':
                # Format: VEGF{GENE:ENSEMBL:ENSG00000112715}
                for entity_text, entity_type, entity_id in matches:
                    if entity_type not in entity_mentions:
                        entity_mentions[entity_type] = []
                    entity_mentions[entity_type].append(f"{entity_text} ({entity_id})")
            
            elif pattern_name == 'inline':
                # Format: [GENE:VEGF:ENSEMBL:ENSG00000112715]
                for entity_type, entity_text, entity_id in matches:
                    if entity_type not in entity_mentions:
                        entity_mentions[entity_type] = []
                    entity_mentions[entity_type].append(f"{entity_text} ({entity_id})")
            
            elif pattern_name == 'simple_bold':
                # Format: **VEGF**
                for entity_text in matches:
                    if 'ENTITY' not in entity_mentions:
                        entity_mentions['ENTITY'] = []
                    entity_mentions['ENTITY'].append(entity_text)
        
        return entity_mentions
    
    def validate_response(self, parsed: ParsedResponse) -> bool:
        """Enhanced validation with entity and content analysis"""
        validation_issues = []
        
        # Check required sections
        required_sections = ['thinking', 'answer', 'explain']
        missing_sections = []
        
        for section in required_sections:
            content = getattr(parsed, section)
            if content is None or content.strip() == "":
                missing_sections.append(section)
        
        if missing_sections:
            validation_issues.append(f"Missing sections: {', '.join(missing_sections)}")
        
        # Validate explain section structure
        if parsed.explain:
            explain_issues = self._validate_explain_section(parsed.explain)
            validation_issues.extend(explain_issues)
        
        # Validate answer section for biomedical content
        if parsed.answer:
            answer_issues = self._validate_answer_section(parsed.answer)
            validation_issues.extend(answer_issues)
        
        # Validate DAG structure
        if parsed.dag:
            dag_issues = self._validate_dag_section(parsed.dag)
            validation_issues.extend(dag_issues)
            
            # Validate DAG consistency with explain section
            consistency_issues = self.validate_dag_consistency(parsed)
            validation_issues.extend(consistency_issues)
        
        parsed.validation_issues = validation_issues
        return len(validation_issues) == 0
    
    def _validate_explain_section(self, explain_section: str) -> List[str]:
        """Validate the explain section structure"""
        issues = []
        lines = explain_section.strip().split('\n')
        
        if not lines:
            issues.append("Empty explain section")
            return issues
        
        # Check for set_context as first line
        first_line = lines[0].strip()
        if not first_line.startswith('set_context'):
            issues.append("Explain section should start with set_context(...)")
        
        # Validate action primitive syntax
        action_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        for line in action_lines:
            if '(' not in line or ')' not in line:
                issues.append(f"Invalid action primitive syntax: {line[:50]}...")
            
            # Check for properly closed parentheses
            if line.count('(') != line.count(')'):
                issues.append(f"Unmatched parentheses in: {line[:50]}...")
        
        return issues
    
    def _validate_answer_section(self, answer_section: str) -> List[str]:
        """Validate answer section for biomedical content quality"""
        issues = []
        
        # Check length
        if len(answer_section.split()) < 20:
            issues.append("Answer section too short (should be substantial)")
        
        if len(answer_section.split()) > 200:
            issues.append("Answer section too long (should be concise)")
        
        # Check for biomedical content indicators
        biomedical_indicators = 0
        for pattern_name, pattern in self.biomedical_action_patterns.items():
            if pattern.search(answer_section):
                biomedical_indicators += 1
        
        if biomedical_indicators < 2:
            issues.append("Answer lacks sufficient biomedical mechanism details")
        
        return issues
    
    def _validate_dag_section(self, dag_section: str) -> List[str]:
        """Validate the DAG section structure"""
        issues = []
        lines = dag_section.strip().split('\n')
        
        if not lines:
            issues.append("Empty DAG section")
            return issues
        
        # Validate DAG edge syntax
        edge_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if not edge_lines:
            issues.append("No valid DAG edges found")
            return issues
        
        for line in edge_lines:
            if not self.dag_edge_pattern.match(line):
                issues.append(f"Invalid DAG edge syntax: {line[:50]}...")
        
        return issues
    
    def extract_dag_edges(self, dag_section: str) -> List[Dict[str, str]]:
        """Extract DAG edges from the DAG section"""
        if not dag_section:
            return []
        
        edges = []
        lines = dag_section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                match = self.dag_edge_pattern.match(line)
                if match:
                    edges.append({
                        'source': match.group(1),
                        'target': match.group(2),
                        'relation': match.group(3)
                    })
        
        return edges
    
    def validate_dag_consistency(self, parsed: ParsedResponse) -> List[str]:
        """Validate consistency between explain section IDs and DAG edges"""
        issues = []
        
        if not parsed.explain or not parsed.dag:
            return issues
        
        # Extract IDs from explain section
        explain_ids = set()
        explain_lines = parsed.explain.strip().split('\n')
        
        for line in explain_lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Look for id="..." pattern
                id_match = re.search(r'id\s*=\s*["\']([^"\']+)["\']', line)
                if id_match:
                    explain_ids.add(id_match.group(1))
        
        # Extract edge references from DAG
        dag_edges = self.extract_dag_edges(parsed.dag)
        dag_ids = set()
        
        for edge in dag_edges:
            dag_ids.add(edge['source'])
            dag_ids.add(edge['target'])
        
        # Check for missing IDs
        missing_in_explain = dag_ids - explain_ids
        if missing_in_explain:
            issues.append(f"DAG references IDs not found in explain section: {', '.join(missing_in_explain)}")
        
        # Check for valid relation types
        valid_relations = {'causal', 'correlative'}
        for edge in dag_edges:
            if edge['relation'] not in valid_relations:
                issues.append(f"Invalid relation type '{edge['relation']}' in DAG edge")
        
        return issues
    
    def extract_action_primitives(self, explain_section: str) -> List[str]:
        """Extract action primitives with improved parsing"""
        if not explain_section:
            return []
        
        lines = explain_section.strip().split('\n')
        actions = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                if '(' in line:
                    # Extract action name more robustly
                    action_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                    if action_match:
                        action_name = action_match.group(1)
                        actions.append(action_name)
        
        return actions
    
    def validate_action_primitives(self, explain_section: str, allowed_primitives: List[str]) -> bool:
        """Enhanced validation of action primitives"""
        if not explain_section:
            return False
        
        extracted_actions = self.extract_action_primitives(explain_section)
        invalid_actions = []
        
        # Convert allowed primitives to set for faster lookup
        allowed_set = set(allowed_primitives)
        
        for action in extracted_actions:
            if action not in allowed_set:
                invalid_actions.append(action)
        
        if invalid_actions:
            logger.warning(f"Invalid action primitives used: {invalid_actions}")
            return False
        
        return True
    
    def clean_explain_section(self, explain_section: str) -> str:
        """Clean and normalize the explain section"""
        if not explain_section:
            return ""
        
        lines = explain_section.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_structured_data(self, explain_section: str, dag_section: str = None) -> Dict:
        """Extract comprehensive structured data from explain and DAG sections"""
        if not explain_section:
            return {}
        
        lines = explain_section.strip().split('\n')
        structured_data = {
            'set_context': None,
            'actions': [],
            'action_count': 0,
            'unique_actions': set(),
            'entities_referenced': set(),
            'relationships': [],
            'dag_edges': [],
            'node_ids': set()
        }
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                if line.startswith('set_context'):
                    structured_data['set_context'] = line
                else:
                    if '(' in line:
                        action_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$', line)
                        if action_match:
                            action_name = action_match.group(1)
                            parameters = action_match.group(2)
                            
                            # Extract ID if present
                            id_match = re.search(r'id\s*=\s*["\']([^"\']+)["\']', parameters)
                            node_id = id_match.group(1) if id_match else None
                            
                            action_data = {
                                'action': action_name,
                                'parameters': parameters,
                                'full_line': line,
                                'id': node_id
                            }
                            
                            structured_data['actions'].append(action_data)
                            structured_data['unique_actions'].add(action_name)
                            
                            if node_id:
                                structured_data['node_ids'].add(node_id)
                            
                            # Extract entity references from parameters
                            entities = self._extract_entities_from_parameters(parameters)
                            structured_data['entities_referenced'].update(entities)
        
        # Process DAG section if provided
        if dag_section:
            dag_edges = self.extract_dag_edges(dag_section)
            structured_data['dag_edges'] = dag_edges
        
        structured_data['action_count'] = len(structured_data['actions'])
        structured_data['unique_actions'] = list(structured_data['unique_actions'])
        structured_data['entities_referenced'] = list(structured_data['entities_referenced'])
        structured_data['node_ids'] = list(structured_data['node_ids'])
        
        return structured_data
    
    def _extract_entities_from_parameters(self, parameters: str) -> Set[str]:
        """Extract entity names from action parameters"""
        entities = set()
        
        # Simple pattern to extract likely entity names (capitalized words)
        entity_pattern = re.compile(r'\b[A-Z][A-Z0-9-]*[A-Z0-9]\b')
        matches = entity_pattern.findall(parameters)
        entities.update(matches)
        
        # Extract quoted strings that might be entity names
        quoted_pattern = re.compile(r'["\']([^"\']+)["\']')
        quoted_matches = quoted_pattern.findall(parameters)
        entities.update(quoted_matches)
        
        return entities
    
    def format_for_display(self, parsed: ParsedResponse) -> str:
        """Enhanced display formatting with entity information"""
        output = []
        
        if parsed.thinking:
            output.append("THINKING:")
            output.append("=" * 50)
            output.append(parsed.thinking)
            output.append("")
        
        if parsed.answer:
            output.append("ANSWER:")
            output.append("=" * 50)
            output.append(parsed.answer)
            output.append("")
        
        if parsed.explain:
            output.append("STRUCTURED EXPLANATION:")
            output.append("=" * 50)
            output.append(parsed.explain)
            output.append("")
        
        if parsed.dag:
            output.append("DAG:")
            output.append("=" * 50)
            output.append(parsed.dag)
            output.append("")
        
        # Add entity information if available
        if parsed.entity_mentions:
            output.append("ENTITY MENTIONS:")
            output.append("=" * 50)
            for entity_type, entities in parsed.entity_mentions.items():
                output.append(f"{entity_type}: {', '.join(entities[:5])}")
            output.append("")
        
        # Add validation issues if any
        if parsed.validation_issues:
            output.append("VALIDATION ISSUES:")
            output.append("=" * 50)
            for issue in parsed.validation_issues:
                output.append(f"- {issue}")
            output.append("")
        
        return '\n'.join(output)
    
    def get_response_quality_score(self, parsed: ParsedResponse) -> float:
        """Calculate a quality score for the response"""
        score = 0.0
        max_score = 12.0  # Increased to account for DAG
        
        # Section completeness (5 points)
        if parsed.thinking:
            score += 1.0
        if parsed.answer:
            score += 1.5
        if parsed.explain:
            score += 1.5
        if parsed.dag:
            score += 1.0  # DAG is now part of completeness
        
        # Content quality (4 points)
        if parsed.answer and len(parsed.answer.split()) >= 30:
            score += 1.0
        if parsed.entity_mentions and len(parsed.entity_mentions) > 0:
            score += 1.0
        if parsed.action_primitives_used and len(parsed.action_primitives_used) >= 3:
            score += 1.0
        if parsed.dag:
            dag_edges = self.extract_dag_edges(parsed.dag)
            if len(dag_edges) >= 2:  # At least 2 edges for meaningful DAG
                score += 1.0
        
        # Validation (3 points)
        if not parsed.validation_issues:
            score += 3.0
        elif len(parsed.validation_issues) <= 2:
            score += 1.5
        
        return score / max_score
    
    def get_dag_statistics(self, parsed: ParsedResponse) -> Dict:
        """Get statistics about the DAG structure"""
        if not parsed.dag:
            return {"has_dag": False}
        
        edges = self.extract_dag_edges(parsed.dag)
        
        # Count relation types
        relation_counts = {}
        for edge in edges:
            relation = edge['relation']
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        # Get unique nodes
        nodes = set()
        for edge in edges:
            nodes.add(edge['source'])
            nodes.add(edge['target'])
        
        return {
            "has_dag": True,
            "num_edges": len(edges),
            "num_nodes": len(nodes),
            "relation_counts": relation_counts,
            "edges": edges
        } 