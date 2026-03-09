#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# User-editable parameters
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT=${CHECKPOINT:-""}          # required: path to .ckpt file
CONFIG_DIR=${CONFIG_DIR:-""}          # optional: dir with config.yaml (auto-detected from ckpt path)
OUTPUT_DIR=${OUTPUT_DIR:-"data/outputs/mdf_eval"}

ISAACGYM_CFG=${ISAACGYM_CFG:-"isaacgym_config_power_plug.yaml"}
GPU=${GPU:-0}
N_ENVS=${N_ENVS:-50}
N_TEST_VIS=${N_TEST_VIS:-2}
MAX_STEPS=${MAX_STEPS:-200}
N_OBS_STEPS=${N_OBS_STEPS:-2}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
HIS_LEN=${HIS_LEN:-4}
SAMPLING_STEPS=${SAMPLING_STEPS:-10}
INFERENCE_MODE=${INFERENCE_MODE:-"policy"}
PRECISION=${PRECISION:-"bf16"}
SEED=${SEED:-100000}

# Path to the force_tool_tactile repo (must be visible inside the container)
FORCE_TOOL_ROOT=${FORCE_TOOL_ROOT:-"/home/zixuanh/force_tool_tactile"}

# Container file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_FILE=${CONTAINER_FILE:-"${REPO_ROOT}/manifeel.sif"}

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "${CHECKPOINT}" ]]; then
    echo "Error: CHECKPOINT is not set."
    echo "Usage: CHECKPOINT=/path/to/epoch=xx.ckpt bash $0"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Build Python args
# ─────────────────────────────────────────────────────────────────────────────
PYTHON_ARGS=(
    "--checkpoint" "${CHECKPOINT}"
    "--output-dir" "${OUTPUT_DIR}"
    "--isaacgym-cfg" "${ISAACGYM_CFG}"
    "--gpu" "${GPU}"
    "--n-envs" "${N_ENVS}"
    "--n-test-vis" "${N_TEST_VIS}"
    "--max-steps" "${MAX_STEPS}"
    "--n-obs-steps" "${N_OBS_STEPS}"
    "--n-action-steps" "${N_ACTION_STEPS}"
    "--his-len" "${HIS_LEN}"
    "--sampling-steps" "${SAMPLING_STEPS}"
    "--inference-mode" "${INFERENCE_MODE}"
    "--precision" "${PRECISION}"
    "--seed" "${SEED}"
)

if [[ -n "${CONFIG_DIR}" ]]; then
    PYTHON_ARGS+=("--config-dir" "${CONFIG_DIR}")
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run inside Apptainer
# ─────────────────────────────────────────────────────────────────────────────
apptainer exec --nv --cleanenv \
    --env LD_PRELOAD= \
    --env GPU="${GPU}" \
    --env FORCE_TOOL_ROOT="${FORCE_TOOL_ROOT}" \
    --bind "${FORCE_TOOL_ROOT}:${FORCE_TOOL_ROOT}" \
    "${CONTAINER_FILE}" bash -ic "
  set -e
  conda activate manifeel
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

  # Add force_tool_tactile packages to PYTHONPATH so MDF model can be imported
  export PYTHONPATH='${FORCE_TOOL_ROOT}:${FORCE_TOOL_ROOT}/third_party/IsaacGymEnvs':\${PYTHONPATH:-}

  cd '${REPO_ROOT}'
  python eval_mdf.py ${PYTHON_ARGS[*]}
"
