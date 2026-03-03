echo "Node Status Check:"
for dir in slurm_logs/*; do
    if [ -d "$dir" ]; then
        # Get Hostname
        HOST=$(grep "Host:" "$dir"/*.out 2>/dev/null | head -n 1 | awk '{print $2}')
        
        # Check for error
        if grep -q "CUDA unknown error" "$dir"/*.err 2>/dev/null; then
            STATUS="FAIL"
        else
            STATUS="OK"
        fi
        
        # Print if we found a host
        if [ ! -z "$HOST" ]; then
            echo "$HOST $STATUS"
        fi
    fi
done | sort | uniq -c