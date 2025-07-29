import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


COMPRESSION_PROMPT = """
Please compress the following biomedical report while preserving all key scientific information, entity mentions, and important relationships. 
Target length: approximately {max_tokens} tokens.

Original report:
{text}

Compressed version:"""
            
@dataclass
class EntitySpan:
    """Named entity span information"""
    begin: int
    end: int
    entity_id: str
    entity_type: str
    text: str = ""
    
    @classmethod
    def from_dict(cls, span_dict: Dict, content: str) -> 'EntitySpan':
        """Create EntitySpan from dictionary"""
        begin = span_dict['begin']
        end = span_dict['end']
        text = content[begin:end] if content else ""
        
        return cls(
            begin=begin,
            end=end,
            entity_id=span_dict['id'],
            entity_type=span_dict['type'],
            text=text
        )

@dataclass
class ProcessedReport:
    """Processed markdown report with entity information"""
    original_text: str
    processed_text: str
    entities: List[EntitySpan]
    total_tokens: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type, 
                    "id": e.entity_id,
                    "begin": e.begin,
                    "end": e.end
                } for e in self.entities
            ],
            "total_tokens": self.total_tokens
        }

class ReportProcessor:
    """Enhanced processor for handling reports with NER annotations"""
    
    def __init__(self, entity_annotation_style: str = "markdown", strategy: str = "enhanced", llm_client=None):
        self.entity_annotation_style = entity_annotation_style
        self.llm_client = llm_client
        self.strategy = strategy
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4
    
    def process_message_with_spans(self, message: Dict[str, Any]) -> Tuple[str, List[EntitySpan]]:
        """Process a message with content and spans"""
        content = message.get('content', '')
        spans_data = message.get('spans', [])
        
        # Create EntitySpan objects
        entities = []
        for span in spans_data:
            entity = EntitySpan.from_dict(span, content)
            entities.append(entity)
        
        # Sort entities by position for proper text enhancement
        entities.sort(key=lambda x: x.begin)
        
        # Enhance text with entity annotations
        processed_text = self.enhance_text_with_entities(content, entities)
        
        return processed_text.strip(), entities
    
    def enhance_text_with_entities(self, content: str, entities: List[EntitySpan]) -> str:
        """Enhance text by adding entity annotations"""
        if not entities:
            return content
        
        enhanced_parts = []
        last_pos = 0
        
        for entity in entities:
            # Add text before entity
            enhanced_parts.append(content[last_pos:entity.begin])
            
            # Add enhanced entity text
            entity_text = content[entity.begin:entity.end]
            enhanced_entity = self.format_entity(entity_text, entity)
            enhanced_parts.append(enhanced_entity)
            
            last_pos = entity.end
        
        # Add remaining text
        enhanced_parts.append(content[last_pos:])
        
        return ''.join(enhanced_parts)
    
    def format_entity(self, text: str, entity: EntitySpan) -> str:
        """Format entity based on annotation style"""
        entity_type = entity.entity_type.upper()
        entity_id = entity.entity_id
        
        if self.entity_annotation_style == "markdown":
            # Format: **VEGF**[GENE:ENSEMBL:ENSG00000112715]
            return f"**{text}**[{entity_type}:{entity_id}]"
        
        elif self.entity_annotation_style == "html":
            # Format: <abbr title="GENE:ENSEMBL:ENSG00000112715" class="gene">VEGF</abbr>
            return f'<abbr title="{entity_type}:{entity_id}" class="{entity_type.lower()}">{text}</abbr>'
        
        elif self.entity_annotation_style == "brackets":
            # Format: VEGF{GENE:ENSEMBL:ENSG00000112715}
            return f"{text}{{{entity_type}:{entity_id}}}"
        
        elif self.entity_annotation_style == "inline":
            # Format: [GENE:VEGF:ENSEMBL:ENSG00000112715]
            return f"[{entity_type}:{text}:{entity_id}]"
        
        else:
            # Default: just bold the entity
            return f"**{text}**"
    
    def process_report(self, report_data: Any) -> ProcessedReport:
        """
        Process a report that may be a string or message with spans
        
        Args:
            report_data: Either a string (markdown) or dict with content/spans
            strategy: Processing strategy - "enhanced" (use as-is with entities) or "compressed" (compress using LLM)
        """
        # Handle different input formats
        if isinstance(report_data, str):
            # Simple markdown string
            processed_text = report_data
            entities = []
        elif isinstance(report_data, dict):
            # Message format with content and spans
            processed_text, entities = self.process_message_with_spans(report_data)
        else:
            raise ValueError("report_data must be string or dict with content/spans")
        
        # Process based on strategy
        if self.strategy == "enhanced":
            # Use the enhanced text as-is with entity annotations
            final_text = processed_text
        elif self.strategy in ["compressed", "llm"]:
            # Extract key findings only when compressing
            final_text = self.compress_with_llm(processed_text, self.llm_client, max_tokens=2000)
        else:
            raise ValueError("strategy must be 'enhanced' or 'compressed'")
        
        return ProcessedReport(
            original_text=report_data.get('content', '') if isinstance(report_data, dict) else report_data,
            processed_text=final_text,
            entities=entities,
            total_tokens=self.estimate_tokens(final_text)
        )
    
    def compress_with_llm(self, text: str, llm_client=None, max_tokens: int = 2000) -> str:
        """
        Compress text using an LLM
        
        Args:
            text: Text to compress
            llm_client: LLM client instance
            max_tokens: Target token count for compressed text
        """
        messages = [{"role": "user", "content": COMPRESSION_PROMPT.format(text=text, max_tokens=max_tokens)}]
        compressed = llm_client.generate(messages)
        return compressed
    
    def create_condensed_report(self, processed_report: ProcessedReport) -> str:
        """Create a condensed version with entity information - now just returns the enhanced text"""
        return processed_report.processed_text 
