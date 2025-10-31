#!/bin/bash
set -e

# ==============================================================================
# Quick Local Test for the `${args_no_hyphens}` Macro
#
# This script runs a simple `echo` command via the wandb agent to inspect
# the exact command-line arguments it generates.
# ==============================================================================

echo "### Starting \`args_no_hyphens\` Inspection Test ###"

# --- Configuration for the Test ---
TEST_ARCH="linear"
TEST_MODE="multi_task/basic"
PROJECT_NAME="astra"

# --- 1. Create a temporary sweep.yaml for the test ---
echo "--- Creating temporary sweep configuration (test_sweep_no_hyphens.yaml)... ---"
TEMP_SWEEP_CONFIG="test_sweep_no_hyphens.yaml"
cat <<EOF > ${TEMP_SWEEP_CONFIG}
# This is the program that the agent will run.
# On Linux/macOS, `echo` is a simple command-line utility.
program: echo

# This is a minimal set of parameters to test with.
parameters:
  architecture.params.dim_1:
    value: 128
  architecture.params.dim_2:
    value: 256

# This is the critical part we are testing.
# The agent should replace \${args_no_hyphens} with the parameters above,
# formatted as `key=value` without any dashes.
command:
  - \${program}
  - "STATIC_ARG_1"
  - \${args_no_hyphens}
  - "STATIC_ARG_2"
EOF

echo "Generated test sweep config:"
cat ${TEMP_SWEEP_CONFIG}
echo "-----------------------------------"

# --- 2. Create the sweep on W&B and get the ID ---
echo "--- Creating test sweep on wandb... ---"
SWEEP_OUTPUT=$(wandb sweep --name "no-hyphens-test-$(date +%s)" --project "$PROJECT_NAME" "${TEMP_SWEEP_CONFIG}" 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create or parse sweep ID." >&2
    exit 1
fi
echo "Successfully created test sweep with ID: ${SWEEP_ID}"

# --- 3. Run the agent and capture its output ---
echo -e "\n### RUNNING THE AGENT (This is the critical test) ###"
# We're running this inside a subshell `()` to easily capture all output.
AGENT_OUTPUT=$(uv run wandb agent --count 1 "${SWEEP_ID}" 2>&1)

echo "--- Full Agent Output ---"
echo "$AGENT_OUTPUT"
echo "-------------------------"

# --- 4. Analyze the output ---
# We're looking for the line where the agent says "About to run command:"
COMMAND_LINE=$(echo "$AGENT_OUTPUT" | grep "About to run command:")

echo -e "\n### ANALYSIS ###"
echo "The command wandb tried to run was:"
echo "$COMMAND_LINE"

# Check if the output contains hyphens where it shouldn't
if echo "$COMMAND_LINE" | grep -q -- "--architecture"; then
    echo -e "\n❌ ❌ ❌  TEST FAILED!  ❌ ❌ ❌"
    echo "The agent generated arguments WITH hyphens, even though we used \`args_no_hyphens\`."
    echo "This confirms the 'pull' method from the Python script is necessary."
else
    echo -e "\n✅ ✅ ✅  TEST PASSED!  ✅ ✅ ✅"
    echo "The agent correctly generated arguments WITHOUT hyphens."
    echo "This suggests the previous failure might have been due to a different issue (e.g., not re-initializing the sweep)."
fi

# --- 5. Cleanup ---
echo "--- Cleaning up temporary test file... ---"
rm ${TEMP_SWEEP_CONFIG}