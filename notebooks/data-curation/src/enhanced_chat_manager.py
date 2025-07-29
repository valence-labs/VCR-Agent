import asyncio
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

from enhanced_chat_client.api.bulk_controller_v1 import (
    bulk_controller_v1_create,
    bulk_controller_v1_export,
    bulk_controller_v1_status,
)
from enhanced_chat_client.models.bulk_conversation_input import BulkConversationInput
from enhanced_chat_client.models.output_style import OutputStyle
from enhanced_chat_client.models.tagged_prompt import TaggedPrompt
from enhanced_chat_client.models.model import Model


@dataclass
class ChatConfiguration:
    """Configuration for enhanced chat"""
    model: Model = Model.CLAUDE_V3_7_SONNET
    max_brave_to_read: int = 0
    max_ci_agent_to_read: int = 0
    brave_enabled: bool = False
    kg_rag_enabled: bool = False
    omim_enabled: bool = False
    google_enabled: bool = False
    pubmed_enabled: bool = False
    fulltext_enabled: bool = False
    pdf_index_enabled: bool = False
    citeline_enabled: bool = False
    preprint_enabled: bool = False
    kg_agent_enabled: bool = False
    max_pubmed_to_read: int = 0
    max_fulltext_to_read: int = 0
    max_pdf_index_to_read: int = 0
    max_citeline_to_read: int = 0
    max_google_to_read: int = 0
    max_preprint_to_read: int = 0
    cov_enabled: bool = False
    output_style: OutputStyle = OutputStyle.CONCISE
    is_private: bool = False
    do_repeats: bool = False

@dataclass
class JobResult:
    """Result from a completed job"""
    job_tag: str
    conversations: List[Any]
    success: bool
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class EnhancedChatManager:
    """Manager for enhanced chat bulk and single operations"""

    def __init__(self, authenticated_client):
        self.client = authenticated_client
        self.config = ChatConfiguration()
        self.active_jobs: Dict[str, Any] = {}
        self.completed_jobs: Dict[str, JobResult] = {}

    def update_config(self, **kwargs):
        """Update the configuration of the chat manager"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def create_prompts(
            self,
            template: str,
            data: List[Dict[str, Any]],
            global_tags: List[str],
            data_tags: Optional[List[str]] = None,
        ) -> List[TaggedPrompt]:
        """Create prompts for bulk job"""
        if global_tags is None:
            global_tags = []
        if data_tags is None:
            data_tags = []
        prompts = [TaggedPrompt(prompt=template.format(**item), tags=global_tags + [tag]) for item, tag in zip(data, data_tags)]
        return prompts

    async def submit_bulk_job(
            self,
            prompts: List[TaggedPrompt],
            job_tag: str,
            config: Optional[ChatConfiguration] = None,
            resubmit: bool = False
        ) -> Any:
        """Submit a bulk job"""
        if config is None:
            config = self.config

        bulk_input = BulkConversationInput(
            prompts=prompts,
            model=config.model,
            max_brave_to_read=config.max_brave_to_read,
            max_ci_agent_to_read=config.max_ci_agent_to_read,
            brave_enabled=config.brave_enabled,
            kg_rag_enabled=config.kg_rag_enabled,
            omim_enabled=config.omim_enabled,
            google_enabled=config.google_enabled,
            pubmed_enabled=config.pubmed_enabled,
            fulltext_enabled=config.fulltext_enabled,
            pdf_index_enabled=config.pdf_index_enabled,
            citeline_enabled=config.citeline_enabled,
            preprint_enabled=config.preprint_enabled,
            kg_agent_enabled=config.kg_agent_enabled,
            max_pubmed_to_read=config.max_pubmed_to_read,
            max_fulltext_to_read=config.max_fulltext_to_read,
            max_pdf_index_to_read=config.max_pdf_index_to_read,
            max_citeline_to_read=config.max_citeline_to_read,
            max_google_to_read=config.max_google_to_read,
            max_preprint_to_read=config.max_preprint_to_read,
            cov_enabled=config.cov_enabled,
            output_style=config.output_style,
            is_private=config.is_private,
            tags=[job_tag],
            do_repeats=config.do_repeats,
        )
        if (job_tag not in self.active_jobs and job_tag not in self.completed_jobs) or resubmit:
            result = await bulk_controller_v1_create.asyncio(client=self.client, body=bulk_input)
            self.active_jobs[job_tag] = {
                'created_response': result,
                'start_time': time.time(),
                'prompts_count': len(prompts),
                "tag": job_tag
            }
        else:
            result = self.active_jobs[job_tag]['created_response']
        self.active_jobs["latest"] = self.active_jobs[job_tag]
        return result

    async def check_job_status(self, job_tag: Optional[str] = None) -> Dict[str, Any]:
        """Check the status of a job"""
        if job_tag is None:
            job_tag = self.active_jobs["latest"]["tag"]

        try:
            status_response = await bulk_controller_v1_status.asyncio(
                client=self.client, 
                body=[job_tag]
            )

            # Check if job exists in our local tracking
            job_in_local = job_tag in self.active_jobs
            total = len(self.active_jobs[job_tag]['created_response'].refs) if job_in_local else 0

            if status_response:
                completed = len(status_response.done)
                failed = len(status_response.failed) if hasattr(status_response, 'failed') else 0
                in_progress = len(status_response.inflight) if hasattr(status_response, 'inflight') else 0
                
                # If job is not in local tracking but has completed items, it might be a completed job
                if not job_in_local and completed > 0:
                    # Try to get the job details from the server
                    try:
                        # We need to reconstruct the job info from the completed refs
                        # This is a fallback for jobs that completed on server but weren't tracked locally
                        return {
                            'job_tag': job_tag,
                            'total': completed + failed + in_progress,
                            'completed': completed,
                            'failed': failed,
                            'in_progress': in_progress,
                            'is_complete': completed > 0 and in_progress == 0,
                            'success_rate': completed / (completed + failed + in_progress) if (completed + failed + in_progress) > 0 else 0,
                            'server_completed': True  # Flag to indicate this was found on server
                        }
                    except Exception as e:
                        logger.warning(f"Could not reconstruct job info for {job_tag}: {e}")
            else:
                completed = failed = in_progress = 0

            return {
                'job_tag': job_tag,
                'total': total,
                'completed': completed,
                'failed': failed,
                'in_progress': in_progress,
                'is_complete': completed == total and total > 0,
                'success_rate': completed / total if total > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error checking status for job {job_tag}: {e}") 
            return {'job_tag': job_tag, 'error': str(e), 'is_complete': False}

    async def wait_for_completion(
            self,
            job_tag: str,
            poll_interval: int = 3,
            timeout: Optional[int] = None,
            show_progress: bool = True
        ) -> bool:
        """Wait for a job to complete"""
        start_time = time.time()
        pbar = None

        try:
            while True:
                status = await self.check_job_status(job_tag)

                if 'error' in status:
                    return False

                if show_progress:
                    total = status.get('total', 0)
                    completed = status.get('completed', 0)

                    if pbar is None and total > 0:
                        pbar = tqdm(total=total, desc=f"Job {job_tag}", unit="tasks")

                    if pbar is not None:
                        pbar.n = completed
                        pbar.refresh()

                if status['is_complete']:
                    if pbar:
                        pbar.close()
                    return True

                if timeout and (time.time() - start_time) > timeout:
                    if pbar:
                        pbar.close()
                    return False

                await asyncio.sleep(poll_interval)

        except Exception as e:
            if pbar:
                pbar.close()
            raise

    async def collect_results(
            self,
            job_tag: str,
            wait_for_completion: bool = True,
            timeout: Optional[int] = None,
            refetch: bool = False
        ) -> JobResult:
        """Collect the results of a job"""
        try:
            # First check if job is already completed locally
            if job_tag in self.completed_jobs and not refetch:
                logger.info(f"Job {job_tag} already completed locally, returning cached results")
                return self.completed_jobs[job_tag]

            # Check if job exists in active jobs
            job_in_active = job_tag in self.active_jobs
            
            if wait_for_completion:
                completed = await self.wait_for_completion(job_tag, timeout=timeout)
                if not completed:
                    # Even if wait_for_completion failed, let's try to collect what we can
                    logger.warning(f"Job {job_tag} did not complete within timeout, attempting to collect partial results")
            
            # Check status to see if job exists on server
            status = await self.check_job_status(job_tag)
            
            if 'error' in status:
                return JobResult(
                    job_tag=job_tag,
                    conversations=[],
                    success=False,
                    error=f"Job status error: {status['error']}",
                    start_time=self.active_jobs.get(job_tag, {}).get('start_time'),
                    end_time=time.time()
                )

            # If job is not in active jobs but has completed items on server
            if not job_in_active and status.get('server_completed', False):
                logger.info(f"Job {job_tag} found completed on server but not tracked locally")
                # We need to get the refs from the status response
                try:
                    status_response = await bulk_controller_v1_status.asyncio(
                        client=self.client, 
                        body=[job_tag]
                    )
                    
                    if status_response and status_response.done:
                        # Extract refs from completed items
                        refs = status_response.done
                        # Check if refs are strings or objects with id attribute
                        if refs and hasattr(refs[0], 'id'):
                            ref_ids = [ref.id for ref in refs]
                        else:
                            # If refs are already strings, use them directly
                            ref_ids = refs
                        
                        export_response = await bulk_controller_v1_export.asyncio(
                            client=self.client,
                            body=ref_ids
                        )
                        
                        result = JobResult(
                            job_tag=job_tag,
                            conversations=export_response,
                            success=True,
                            start_time=None,  # We don't have start time for server-completed jobs
                            end_time=time.time()
                        )
                        
                        self.completed_jobs[job_tag] = result
                        return result
                    else:
                        return JobResult(
                            job_tag=job_tag,
                            conversations=[],
                            success=False,
                            error="Job found on server but no completed items available",
                            end_time=time.time()
                        )
                except Exception as e:
                    logger.error(f"Error collecting server-completed job {job_tag}: {e}")
                    return JobResult(
                        job_tag=job_tag,
                        conversations=[],
                        success=False,
                        error=f"Error collecting server-completed job: {e}",
                        end_time=time.time()
                    )

            # Standard flow for jobs tracked locally
            if not job_in_active:
                return JobResult(
                    job_tag=job_tag,
                    conversations=[],
                    success=False,
                    error="Job not found in active jobs and not completed on server"
                )

            created_response = self.active_jobs[job_tag]['created_response']

            export_response = await bulk_controller_v1_export.asyncio(
                client=self.client,
                body=[ref.id for ref in created_response.refs]
            )

            job_start_time = self.active_jobs[job_tag].get('start_time')
            result = JobResult(
                job_tag=job_tag,
                conversations=export_response,
                success=True,
                start_time=job_start_time,
                end_time=time.time()
            )

            self.completed_jobs[job_tag] = result
            del self.active_jobs[job_tag]

            return result

        except Exception as e:
            error_msg = f"Error collecting results for job {job_tag}: {e}"
            logger.error(error_msg)

            return JobResult(
                job_tag=job_tag,
                conversations=[],
                success=False,
                error=error_msg,
                start_time=self.active_jobs.get(job_tag, {}).get('start_time'),
                end_time=time.time()
            )

    async def submit_and_collect(
        self,
        prompts: Union[TaggedPrompt, List[TaggedPrompt]],
        job_tag: str,
        config: Optional[ChatConfiguration] = None,
        timeout: Optional[int] = None,
        resubmit: bool = False
    ) -> JobResult:
        """Submit a bulk job and collect the results"""
        if isinstance(prompts, TaggedPrompt):
            prompts = [prompts]

        await self.submit_bulk_job(prompts, job_tag, config, resubmit=resubmit)
        logger.info(f"Job {job_tag} submitted")
        output = await self.collect_results(job_tag=job_tag, wait_for_completion=True, timeout=timeout)
        logger.info(f"Job {job_tag} completed")
        return output

    def get_job_summary(self) -> Dict[str, Any]:
        """Get a summary of the job status"""
        return {
            'active_jobs': list(self.active_jobs.keys()),
            'completed_jobs': list(self.completed_jobs.keys()),
            'total_active': len(self.active_jobs),
            'total_completed': len(self.completed_jobs)
        }

    def extract_messages(self, conversations: List[Any]) -> List[Dict[str, Any]]:
        messages = []

        for conversation in conversations:
            try:
                conversation_data = {
                    'id': conversation.id,
                    'timestamp': conversation.timestamp,
                    'user': conversation.user,
                    'tags': conversation.tags,
                    'is_private': conversation.is_private,
                    'messages': []
                }

                for message in conversation.messages:
                    message_data = {
                        'content': message.content,
                        'role': str(message.role),
                        'spans': [
                            {
                                'begin': span.begin,
                                'end': span.end,
                                'id': span.id,
                                'type': str(span.type_)
                            }
                            for span in message.spans
                        ] if hasattr(message, 'spans') and message.spans else []
                    }

                    if hasattr(message, 'stats') and message.stats:
                        message_data['stats'] = {
                            'llm_calls': message.stats.llm_calls,
                            'documents_read': message.stats.documents_read,
                            'time_elapsed_seconds': message.stats.time_elapsed_seconds
                        }

                    conversation_data['messages'].append(message_data)

                messages.append(conversation_data)

            except Exception as e:
                logger.error(f"Error extracting message from conversation: {e}")
                continue

        return messages 
