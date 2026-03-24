#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
SEED=${SEED:-44}
NUM_DEMOS=${NUM_DEMOS:-50}
NUM_EPOCH=${NUM_EPOCH:-1000}

DATASET_PATH=${DATASET_PATH:-data/usb_quan_Aug05}
ISAACGYM_CONFIG=${ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}

TASK_NAME=${TASK_NAME:-vision_wrist}  # vision_wrist | vistac_wrist | visff_wrist | vision_front | vistac_front | visff_front
INPUT_TYPE=${INPUT_TYPE:-vision}   # vision | vistac | tacff
ENV_TAG=${ENV_TAG:-usb_wrist_0805}
LOG_NAME=${LOG_NAME:-dp_usb_tacff}

# GPU device index (0-based physical index; used for IsaacGym graphics and CUDA_VISIBLE_DEVICES)
GPU=${GPU:-0}

# Optional Hydra overrides
IMAGENET_NORM=${IMAGENET_NORM:-false}   # set true for tacff runs if needed
ACTION_SHAPE=${ACTION_SHAPE:-}          # set to 7 for gripper tasks, leave empty otherwise

# Transformer architecture params (override via env vars if needed)
N_LAYER=${N_LAYER:-8}
N_HEAD=${N_HEAD:-4}
N_EMB=${N_EMB:-256}
P_DROP_ATTN=${P_DROP_ATTN:-0.3}

# Vision backbone: "resnet18" (default) or a timm model name like "vit_tiny_patch16_224.augreg_in21k"
BACKBONE=${BACKBONE:-resnet18}

# Container file (default assumes you run from repo root and built it there)
RUN_NAME=${RUN_NAME:-}

# Container file (default assumes you run from repo root and built it there)
CONTAINER_FILE=${CONTAINER_FILE:-manifeel.sif}

# -------------------------
# Derived names
# -------------------------
EXP_NAME="DiT_${INPUT_TYPE}_${ENV_TAG}_${NUM_DEMOS}"
if [[ -n "${RUN_NAME}" ]]; then
  EXP_NAME="${EXP_NAME}_${RUN_NAME}"
fi
OUT_DIR="data/outputs/${EXP_NAME}/${SEED}"

# Repo root resolution (script works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Build Hydra args
# -------------------------
HYDRA_ARGS=(
  "--config-name=train_diffusion_transformer_hybrid_workspace.yaml"
  "task=${TASK_NAME}"
  "exp_name=${EXP_NAME}"
  "dataset_path=${DATASET_PATH}"
  "isaacgym_cfg_name=${ISAACGYM_CONFIG}"
  "training.seed=${SEED}"
  "training.num_epochs=${NUM_EPOCH}"
  "training.device=cuda:${GPU}"
  "task.dataset.max_train_episodes=${NUM_DEMOS}"
  "hydra.run.dir=${OUT_DIR}"
  "logging.project=${LOG_NAME}"
  "policy.n_layer=${N_LAYER}"
  "policy.n_head=${N_HEAD}"
  "policy.n_emb=${N_EMB}"
  "policy.p_drop_attn=${P_DROP_ATTN}"
)

if [[ "${IMAGENET_NORM}" == "true" ]]; then
  HYDRA_ARGS+=("policy.obs_encoder.imagenet_norm=True")
fi

if [[ -n "${ACTION_SHAPE}" ]]; then
  HYDRA_ARGS+=("task.shape_meta.action.shape=[${ACTION_SHAPE}]")
fi

if [[ "${BACKBONE}" != "resnet18" ]]; then
  HYDRA_ARGS+=(
    "policy.obs_encoder.rgb_model._target_=diffusion_policy.model.vision.model_getter.get_timm_model"
    "policy.obs_encoder.rgb_model.name=${BACKBONE}"
    "+policy.obs_encoder.rgb_model.pretrained=true"
    "policy.obs_encoder.use_group_norm=False"
    "policy.obs_encoder.resize_shape=[224,224]"
  )
fi

# -------------------------
# Run inside Apptainer
# -------------------------
apptainer exec --nv --cleanenv --env LD_PRELOAD= --env GPU="${GPU}" "${REPO_ROOT}/${CONTAINER_FILE}" bash -ic "
  set -e
  conda activate manifeel
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  cd '${REPO_ROOT}'
  python train.py ${HYDRA_ARGS[*]}
"
