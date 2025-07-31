import asyncio
import time


async def process_treatments_efficiently(treatments, chat_manager, report_template, base_job_tag, chunk_size=20):
    """
    Efficiently process treatments in parallel chunks with proper job management
    """

    async def process_chunk(chunk_data):
        """Process a single chunk"""
        chunk, chunk_idx = chunk_data

        # Create unique job tag for this chunk
        job_tag = f"{base_job_tag}-chunk-{chunk_idx}"

        try:
            # Check if job already exists and is completed
            status = await chat_manager.check_job_status(job_tag)

            if status.get("is_complete", False):
                print(f"✓ Chunk {chunk_idx} already completed, collecting existing results...")
                output = await chat_manager.collect_results(job_tag=job_tag, wait_for_completion=False)
                return {"chunk_index": chunk_idx, "job_tag": job_tag, "output": output, "status": "reused_existing"}

            # Create prompts for this chunk
            prompts = chat_manager.create_prompts(
                template=report_template,
                data=[{"treatment": treatment} for treatment in chunk],
                global_tags=["hooke-explain", "case-study", "perturbation"],
                data_tags=[
                    "{}-{}".format(treatment["perturbation"]["name"], treatment["context"]["disease_model"])
                    for treatment in chunk
                ],
            )

            print(f"🚀 Launching chunk {chunk_idx} with {len(chunk)} treatments...")

            # Submit and collect results
            output = await chat_manager.submit_and_collect(
                prompts=prompts,
                job_tag=job_tag,
                timeout=600,  # 10 minute timeout per chunk
                resubmit=False,  # Don't resubmit if job exists
            )

            return {
                "chunk_index": chunk_idx,
                "job_tag": job_tag,
                "output": output,
                "prompts": prompts,
                "status": "newly_processed",
            }

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_idx}: {e}")
            return {"chunk_index": chunk_idx, "job_tag": job_tag, "output": None, "error": str(e), "status": "failed"}

    # Split treatments into chunks with their indices
    chunks_with_indices = [(treatments[i : i + chunk_size], i) for i in range(0, len(treatments), chunk_size)]

    print(f"📊 Processing {len(treatments)} treatments in {len(chunks_with_indices)} parallel chunks...")

    # Process all chunks in parallel
    start_time = time.time()

    # Use semaphore to limit concurrent jobs if needed
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent chunks

    async def process_with_semaphore(chunk_data):
        async with semaphore:
            return await process_chunk(chunk_data)

    # Run all chunks in parallel
    results = await asyncio.gather(
        *[process_with_semaphore(chunk_data) for chunk_data in chunks_with_indices], return_exceptions=True
    )

    end_time = time.time()

    # Process results
    successful_results = []
    failed_results = []
    reused_results = []

    for result in results:
        if isinstance(result, Exception):
            failed_results.append({"error": str(result), "status": "exception"})
        elif result["status"] == "failed":
            failed_results.append(result)
        elif result["status"] == "reused_existing":
            reused_results.append(result)
        else:
            successful_results.append(result)

    # Print summary
    print("\n📈 Processing Summary:")
    print(f"   ⏱️  Total time: {end_time - start_time:.1f} seconds")
    print(f"   ✅ Successful: {len(successful_results)}")
    print(f"   🔄 Reused existing: {len(reused_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")
    print(f"   📊 Total chunks: {len(chunks_with_indices)}")

    if failed_results:
        print("\n❌ Failed chunks:")
        for failed in failed_results:
            print(f"   - Chunk {failed.get('chunk_index', 'unknown')}: {failed.get('error', 'unknown error')}")

    return {
        "successful": successful_results,
        "reused": reused_results,
        "failed": failed_results,
        "total_time": end_time - start_time,
        "all_results": successful_results + reused_results,  # Combined successful results
    }
