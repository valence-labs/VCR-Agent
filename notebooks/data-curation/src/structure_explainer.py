"""
Improved StructureExplainer for generating structured biomedical explanations
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm as atqdm

from .llm_client import LLMClient, create_llm_client
from .response_parser import ResponseParser, ParsedResponse
from .report_processor import ReportProcessor

_BASE_DIR = Path(__file__).parent.parent

class StructureExplainer:
    """Improved class for generating structured biomedical explanations"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        report_processing_strategy: str = "enhanced",
        template_path: Optional[str] = _BASE_DIR / "templates" / "structure-explain.txt",
        action_primitives_path: Optional[str] = _BASE_DIR / "action_primitives.json",
    ):
        self.llm_client = llm_client
        self.parser = ResponseParser()
        self.report_processor = ReportProcessor(
            strategy=report_processing_strategy, 
            llm_client=llm_client
        )
        
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
        with open(action_primitives_path, 'r', encoding='utf-8') as f:
            self.action_primitives = json.load(f)
        
        self.allowed_actions = [ap["action"] for ap in self.action_primitives]

    async def generate_explanation(
        self,
        question: str,
        report: str,
        validate_response: bool = True,
        max_retries: int = 3
    ) -> ParsedResponse:
        """Generate structured explanation for a single request"""
        
        processed_report = self.report_processor.process_report(report)
        prompt = self.template.format(
            action_primitives=self.action_primitives,
            question=question,
            report=processed_report.processed_text
        )
        
        for attempt in range(max_retries + 1):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm_client.agenerate(messages)
                parsed = self.parser.parse_response(response)
                
                if validate_response:
                    if not self.parser.validate_response(parsed):
                        if attempt < max_retries:
                            logger.warning(f"Validation failed on attempt {attempt + 1}, retrying...")
                            continue
                    
                    if parsed.explain:
                        if not self.parser.validate_action_primitives(parsed.explain, self.allowed_actions):
                            if attempt < max_retries:
                                logger.warning(f"Action primitive validation failed on attempt {attempt + 1}, retrying...")
                                continue
                
                return parsed
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Generation failed on attempt {attempt + 1}: {e}, retrying...")
                    continue
                else:
                    logger.error(f"Error in async generation: {e}")
                    raise
        
        raise RuntimeError("Failed to generate valid explanation after retries")

    async def process_batch(
        self,
        data: List[Dict[str, str]],
        question_key: str = "question",
        report_key: str = "report",
        id_column: Optional[str] = None,
        validate_responses: bool = True,
        max_concurrent: int = 5,  # Consistent with llm_client.py default
        chunk_size: Optional[int] = None,  # Auto-chunking for very large batches
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process batch of question-report pairs efficiently
        
        Args:
            data: List of dictionaries containing questions and reports
            question_key: Key for question in each data item
            report_key: Key for report in each data item
            id_column: Optional ID column to preserve
            validate_responses: Whether to validate responses
            max_concurrent: Maximum concurrent requests (consistent with LLM client default)
            chunk_size: Size of chunks for very large batches (auto-calculated if None)
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with results
        """
        
        total_items = len(data)
        logger.info(f"Processing {total_items} items with max_concurrent={max_concurrent}")
        
        # Auto-calculate chunk size for very large batches (>200 items)
        if chunk_size is None:
            if total_items > 200:
                chunk_size = min(100, total_items // 4)  # Process in chunks of 100 or 1/4 of total
            else:
                chunk_size = total_items  # Process all at once for smaller batches
        
        # Process in chunks if needed
        if total_items > chunk_size:
            return await self._process_large_batch_in_chunks(
                data, question_key, report_key, id_column, validate_responses,
                max_concurrent, chunk_size, show_progress
            )
        
        # For smaller batches, use the LLM client's batch processing when possible
        if total_items <= 50 and not validate_responses:
            return await self._process_with_llm_batch(
                data, question_key, report_key, id_column, max_concurrent, show_progress
            )
        
        # Regular batch processing for medium-sized batches or when validation is needed
        return await self._process_single_batch(
            data, question_key, report_key, id_column, validate_responses,
            max_concurrent, show_progress
        )

    async def _process_with_llm_batch(
        self,
        data: List[Dict[str, str]],
        question_key: str,
        report_key: str,
        id_column: Optional[str],
        max_concurrent: int,
        show_progress: bool
    ) -> pd.DataFrame:
        """Use LLM client's batch processing for simple cases (consistent with llm_client.py)"""
        
        # Prepare batch requests for LLM client
        batch_requests = []
        for i, item in enumerate(data):
            question = item[question_key]
            report = item[report_key]
            
            processed_report = self.report_processor.process_report(report)
            prompt = self.template.format(
                action_primitives=self.action_primitives,
                question=question,
                report=processed_report.processed_text
            )
            
            messages = [{"role": "user", "content": prompt}]
            batch_requests.append({"messages": messages, "index": i})
        
        try:
            # Use LLM client's batch processing
            responses = await self.llm_client.agenerate_batch(
                batch_requests, max_concurrent=max_concurrent
            )
            
            # Process responses
            results = []
            for i, (item, response) in enumerate(zip(data, responses)):
                try:
                    if response.startswith("Error:"):
                        raise Exception(response[6:])  # Remove "Error:" prefix
                    
                    parsed = self.parser.parse_response(response)
                    
                    result = {
                        "index": i,
                        "question": item[question_key],
                        "thinking": parsed.thinking,
                        "answer": parsed.answer,
                        "explain": parsed.explain,
                        "dag": parsed.dag,
                        "raw_response": parsed.raw_response,
                        "success": True,
                        "error": None
                    }
                    
                    if id_column and id_column in item:
                        result["id"] = item[id_column]
                    
                    # Add other input fields
                    for key, value in item.items():
                        if key not in [question_key, report_key, id_column]:
                            result[f"input_{key}"] = value
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing LLM batch response {i}: {e}")
                    results.append({
                        "index": i,
                        "question": item.get(question_key, ""),
                        "thinking": None,
                        "answer": None,
                        "explain": None,
                        "dag": None,
                        "raw_response": "",
                        "success": False,
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in LLM batch processing: {e}")
            # Fallback to regular processing
            return await self._process_single_batch(
                data, question_key, report_key, id_column, True, max_concurrent, show_progress
            )

    async def _process_single_batch(
        self,
        data: List[Dict[str, str]],
        question_key: str,
        report_key: str,
        id_column: Optional[str],
        validate_responses: bool,
        max_concurrent: int,
        show_progress: bool
    ) -> pd.DataFrame:
        """Process a single batch without chunking"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_item(i: int, item: Dict[str, str]) -> Dict:
            async with semaphore:
                try:
                    question = item[question_key]
                    report = item[report_key]
                    
                    parsed = await self.generate_explanation(
                        question=question,
                        report=report,
                        validate_response=validate_responses
                    )
                    
                    result = {
                        "index": i,
                        "question": question,
                        "thinking": parsed.thinking,
                        "answer": parsed.answer,
                        "explain": parsed.explain,
                        "dag": parsed.dag,
                        "raw_response": parsed.raw_response,
                        "success": True,
                        "error": None
                    }
                    
                    if id_column and id_column in item:
                        result["id"] = item[id_column]
                    
                    # Add other input fields
                    for key, value in item.items():
                        if key not in [question_key, report_key, id_column]:
                            result[f"input_{key}"] = value
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    return {
                        "index": i,
                        "question": item.get(question_key, ""),
                        "thinking": None,
                        "answer": None,
                        "explain": None,
                        "dag": None,
                        "raw_response": "",
                        "success": False,
                        "error": str(e)
                    }
        
        # Create tasks for all items
        tasks = [process_single_item(i, item) for i, item in enumerate(data)]
        
        # Run with progress bar
        if show_progress:
            results = []
            for coro in atqdm.as_completed(tasks, desc="Processing explanations"):
                result = await coro
                results.append(result)
            # Sort back to original order
            results.sort(key=lambda x: x['index'])
        else:
            results = await asyncio.gather(*tasks)
        
        return pd.DataFrame(results)

    async def _process_large_batch_in_chunks(
        self,
        data: List[Dict[str, str]],
        question_key: str,
        report_key: str,
        id_column: Optional[str],
        validate_responses: bool,
        max_concurrent: int,
        chunk_size: int,
        show_progress: bool
    ) -> pd.DataFrame:
        """Process very large batches in chunks to manage memory and rate limits"""
        
        total_items = len(data)
        chunks = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        logger.info(f"Processing {total_items} items in {len(chunks)} chunks of size {chunk_size}")
        
        all_results = []
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)")
            
            # Adjust indices for this chunk
            chunk_with_adjusted_indices = []
            for i, item in enumerate(chunk):
                adjusted_item = item.copy()
                chunk_with_adjusted_indices.append(adjusted_item)
            
            # Process this chunk
            chunk_results = await self._process_single_batch(
                chunk_with_adjusted_indices, question_key, report_key, id_column,
                validate_responses, max_concurrent, show_progress and chunk_idx == 0  # Only show progress for first chunk
            )
            
            # Adjust indices back to global indices
            chunk_results['index'] = chunk_results['index'] + (chunk_idx * chunk_size)
            
            all_results.append(chunk_results)
            
            # Brief pause between chunks to be respectful to the API
            if chunk_idx < len(chunks) - 1:  # Don't sleep after the last chunk
                await asyncio.sleep(1)
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results = combined_results.sort_values('index').reset_index(drop=True)
        
        logger.info(f"Completed processing {total_items} items in chunks")
        return combined_results

    async def process_from_dataframe(
        self,
        df: pd.DataFrame,
        question_key: str = "question",
        report_key: str = "report",
        **kwargs
    ) -> pd.DataFrame:
        """Process from pandas DataFrame"""
        data = df.to_dict('records')
        return await self.process_batch(data=data, question_key=question_key, report_key=report_key, **kwargs)

    def get_statistics(self, results_df: pd.DataFrame) -> Dict:
        """Get processing statistics"""
        stats = {
            "total_processed": len(results_df),
            "successful": results_df['success'].sum(),
            "failed": (~results_df['success']).sum(),
            "success_rate": results_df['success'].mean(),
        }
        
        # Action primitive usage
        if 'explain' in results_df.columns:
            action_usage = {}
            for explain_text in results_df['explain'].dropna():
                actions = self.parser.extract_action_primitives(explain_text)
                for action in actions:
                    action_usage[action] = action_usage.get(action, 0) + 1
            
            stats["action_primitive_usage"] = action_usage
            stats["most_used_actions"] = sorted(
                action_usage.items(), key=lambda x: x[1], reverse=True
            )[:10]
        
        return stats

    # Resource management methods (consistent with llm_client.py)
    async def aclose(self):
        """Close async resources"""
        await self.llm_client.aclose()

    def close(self):
        """Close sync resources"""
        self.llm_client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.aclose()


def create_structure_explainer(model: str = "claude-sonnet-4@20250514", **kwargs) -> StructureExplainer:
    """Create improved StructureExplainer"""
    llm_client = create_llm_client(model=model)
    return StructureExplainer(llm_client, **kwargs) 
