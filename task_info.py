"""
Central registry of per-task metadata for ManiFeel tasks.

Used by:
  - launchers/launch_main_df_manifeel*.py  (training config)
  - manifeel/policy/run_mmdf_eval.py       (eval, inside container)

Each entry maps a task key to its metadata dict.  The task key is an
arbitrary short name used in the launchers.

Fields
------
action_dim   : int   – dimensionality of the action vector
state_dim    : int   – dimensionality of the proprioceptive state vector
dataset_name : str   – raw_datasets/<name> sub-folder (matches --name in convert_to_vistac_h5.py)
task_cfg     : str   – manifeel runner/shape_meta yaml stem (--task-cfg in run_mmdf_eval.py)
isaacgym_cfg : str   – IsaacGym task config yaml filename (--isaacgym-cfg in run_mmdf_eval.py)

Note: task_cfg always points to a vistac-style yaml with shape_meta (e.g. vistac_wrist).
      isaacgym_cfg points to the TacSL env config (e.g. isaacgym_config_power_plug.yaml).
      For 7-DOF tasks the action_dim here overrides shape_meta.action.shape at runtime.
"""

TASK_INFO = {
    # ── plug insertion (power plug, 6-DOF) ─────────────────────────────────
    "plug": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "plug_quan_Aug02",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_power_plug.yaml",
        "max_steps":    500,
    },
    # ── ball sorting (7-DOF) ────────────────────────────────────────────────
    "sorting": {
        "action_dim":   7,
        "state_dim":    7,
        "dataset_name": "sorting_quan_Aug8",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_ball_sorting.yaml",
        "max_steps":    500,
    },
    # ── USB insertion (6-DOF) ───────────────────────────────────────────────
    "usb": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "usb_quan_Aug05",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_usb.yaml",
        "max_steps":    500,
    },
    # ── bulb insertion (7-DOF) ──────────────────────────────────────────────
    "bulb": {
        "action_dim":   7,
        "state_dim":    7,
        "dataset_name": "bulb_quan_Sep19",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_bulb.yaml",
        "max_steps":    500,
    },
    # ── gear meshing (6-DOF) ────────────────────────────────────────────────
    "gear": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "gear_quan_Sep15",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_gear.yaml",
        "max_steps":    500,
    },
    # ── nut-bolt (7-DOF) ────────────────────────────────────────────────────
    "nutbolt": {
        "action_dim":   7,
        "state_dim":    7,
        "dataset_name": "nutbolt_quan_July1",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_nutbolt.yaml",
        "max_steps":    500,
    },
    # ── peg-in-hole (6-DOF) ─────────────────────────────────────────────────
    "pih": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "pih_quan_June06",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_pih.yaml",
        "max_steps":    500,
    },
    # ── object search / explore (6-DOF) ─────────────────────────────────────
    "explore": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "explore_quan_June17",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_explore.yaml",
        "max_steps":    500,
    },
    # ── blind insertion (6-DOF) ──────────────────────────────────────────────
    "blindinsert": {
        "action_dim":   6,
        "state_dim":    7,
        "dataset_name": "blindinsert_quan_Aug15",
        "task_cfg":     "vistac_wrist",
        "isaacgym_cfg": "isaacgym_config_blindinsert.yaml",
        "max_steps":    500,
    },
}

# Reverse lookup: isaacgym_cfg filename → task key (unique per task)
_ISAACGYM_CFG_TO_KEY = {v["isaacgym_cfg"]: k for k, v in TASK_INFO.items()}


def get_by_isaacgym_cfg(isaacgym_cfg_name: str) -> dict:
    """Return the TASK_INFO entry for a given --isaacgym-cfg filename."""
    key = _ISAACGYM_CFG_TO_KEY.get(isaacgym_cfg_name)
    if key is None:
        raise KeyError(
            f"Unknown isaacgym_cfg '{isaacgym_cfg_name}'. "
            f"Add it to manifeel/manifeel/task_info.py."
        )
    return TASK_INFO[key]
