import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List


def _run_job_with_error_handling(job_function: Callable, args: tuple) -> Dict[str, Any]:
    """Wrapper that handles errors and timing"""
    config, job_id = args
    try:
        start_time = time.time()
        result = job_function(config)

        return {
            "job_id": job_id,
            "config": config,
            "status": "success",
            "result": result,
            "duration": time.time() - start_time,
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "config": config,
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }


def run_hyperparameter_search(
    job_function: Callable[[Dict[str, Any]], Any],
    job_configs: List[Dict[str, Any]],
    results_dir: str = "hyperparameter_results",
) -> Dict[str, int]:
    """
    Run hyperparameter search given a job function and a list of configurations. Schedules jobs in parallel and saves
    results to disk.

    Your job function should take a config dict and return a result dict.

    Args:
        job_function: Function that takes a config dict and returns results
        job_configs: List of configuration dictionaries
        results_dir: Directory to save results

    Returns:
        Summary stats: {'completed': N, 'failed': N, 'total': N}
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    total_jobs = len(job_configs)
    max_workers = os.cpu_count()

    print(f"Running {total_jobs} jobs on {max_workers} cores...")

    completed = failed = skipped = 0

    def save_result(result: Dict[str, Any]) -> None:
        """Save result to disk immediately"""
        job_id = result["job_id"]
        file_path = results_path / f"job_{job_id:06d}.json"

        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)

    def job_already_exists(job_id: int) -> bool:
        """Check if job result already exists"""
        file_path = results_path / f"job_{job_id:06d}.json"
        return file_path.exists()

    # Create a partial function that includes the job_function
    worker_function = partial(_run_job_with_error_handling, job_function)

    # Filter out jobs that already exist
    jobs_to_run = []
    for job_id, config in enumerate(job_configs):
        if job_already_exists(job_id):
            skipped += 1
            print(f"⏭ Job {job_id} file already exists, skipping ({skipped} skipped)")
        else:
            jobs_to_run.append((job_id, config))

    if not jobs_to_run:
        print("All jobs already completed!")
        return {"completed": skipped, "failed": 0, "total": total_jobs, "skipped": skipped}

    print(f"Running {len(jobs_to_run)} remaining jobs (skipped {skipped})")

    # Submit only new jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # maps futures to job ids
        future_to_job = {executor.submit(worker_function, (config, job_id)): job_id for job_id, config in jobs_to_run}

        # Process results as they complete
        for future in as_completed(future_to_job):
            result = future.result()
            save_result(result)

            if result["status"] == "success":
                completed += 1
                print(f"✓ Job {result['job_id']} completed ({completed + failed + skipped}/{total_jobs})")
            else:
                failed += 1
                print(
                    f"✗ Job {result['job_id']} failed: {result['error']} ({completed + failed + skipped}/{total_jobs})"
                )

    # Save summary
    summary = {"completed": completed + skipped, "failed": failed, "total": total_jobs, "skipped": skipped}
    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Complete! {completed + skipped} succeeded ({skipped} skipped), {failed} failed")
    return summary


# Example usage:
def my_test_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple test job that sleeps for a given amount of time

    It fails if the wait time is exactly 5.
    """
    wait_time = config["wait_time"]

    if wait_time == 5:
        raise Exception("Wait time is 5")

    time.sleep(wait_time)
    return {"wait_time": wait_time}


def example_usage():
    times = [random.randint(1, 10) for _ in range(100)]
    configs = [{"wait_time": time} for time in times]

    run_hyperparameter_search(my_test_job, configs)
