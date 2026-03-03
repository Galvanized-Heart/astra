import os
import glob
import re

# --- CONFIGURATION ---
LOG_DIR = "slurm_logs"
JOB_PREFIX = "21"  # Only look at these jobs

def get_file_content(filepath):
    """Reads file content safely."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def parse_job(job_dir):
    job_id = os.path.basename(job_dir)
    
    # Locate files
    out_files = glob.glob(os.path.join(job_dir, "*.out"))
    err_files = glob.glob(os.path.join(job_dir, "*.err"))
    
    if not out_files or not err_files:
        return None

    # We assume one pair per folder based on your description
    out_content = get_file_content(out_files[0])
    err_content = get_file_content(err_files[0])

    # --- EXTRACT METADATA ---
    # Find Host
    host_match = re.search(r"^Host:\s+(.*)", out_content, re.MULTILINE)
    host = host_match.group(1).strip() if host_match else "Unknown"

    # Find Start Time
    time_match = re.search(r"^Start time:\s+(.*)", out_content, re.MULTILINE)
    start_time = time_match.group(1).strip() if time_match else "Unknown"

    # --- DETERMINE STATUS ---
    status = "UNKNOWN"
    details = "-"

    # Priority 1: OOM (Out of Memory) - System level kill
    if "oom_kill" in err_content or "OutOfMemoryError" in err_content:
        status = "OOM KILL"
        details = "Batch size too large"

    # Priority 2: Hardware Failure (CUDA Unknown)
    elif "CUDA unknown error" in err_content:
        status = "CUDA FAIL"
        details = "Node Hardware Error"

    # Priority 3: Network/Socket Error (HuggingFace download storm)
    elif "Device or resource busy" in err_content or "NewConnectionError" in err_content:
        status = "NET ERROR"
        details = "HF Connection Failed"

    # Priority 4: Standard Python Error (Code bug)
    elif "Traceback" in err_content:
        status = "CODE FAIL"
        # Try to grab the last line of the traceback for context
        lines = err_content.splitlines()
        details = lines[-1][:50] if lines else "Python Error"

    # Priority 5: Success
    # We check the .out file for the specific completion string you provided
    elif "Training complete!" in out_content:
        status = "SUCCESS"
        details = "Finished successfully"
    
    # Fallback: If it's still running or just started
    elif "Starting SLURM Job" in out_content:
        status = "RUNNING?"
        details = "No errors yet"

    return {
        "id": job_id,
        "host": host,
        "date": start_time,
        "status": status,
        "details": details
    }

# --- MAIN EXECUTION ---
print(f"{'JOB ID':<10} | {'HOST':<10} | {'STATUS':<10} | {'DATE':<25} | {'DETAILS'}")
print("-" * 90)

job_dirs = glob.glob(os.path.join(LOG_DIR, f"{JOB_PREFIX}*"))
results = []

for d in sorted(job_dirs):
    if os.path.isdir(d):
        res = parse_job(d)
        if res:
            results.append(res)
            print(f"{res['id']:<10} | {res['host']:<10} | {res['status']:<10} | {res['date']:<25} | {res['details']}")

# --- SUMMARY ---
print("-" * 90)
print("SUMMARY:")
counts = {}
for r in results:
    s = r['status']
    counts[s] = counts.get(s, 0) + 1

for k, v in counts.items():
    print(f"{k}: {v}")