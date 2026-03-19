"""
Socket-based policy client for ManiFeel env runner.

Runs inside the Apptainer container (Python 3.8 / IsaacGym).
Connects to the general MMDF policy server (physical/mmdf_policy_server.py)
running in the isaac_tac environment.

Implements BaseImagePolicy so it can be dropped into ManifeelRunner.run().

This client is responsible for mapping ManiFeel observation keys to MMDF
modality names before sending to the server:
  right_tactile_camera_taxim  ->  taxim   (B, T, C, H, W) float32 in [0,1]
  wrist                      ->  wrist   (B, T, C, H, W) float32 in [0,1]
  state                      ->  state   (B, T, 7)
"""

import pickle
import socket
import struct
import time

import numpy as np
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


# ── Wire helpers (must match physical/mmdf_policy_server.py) ─────────────────

def _send(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def _recv(sock):
    raw_len = _recvall(sock, 4)
    if raw_len is None:
        raise ConnectionError("Server disconnected")
    (length,) = struct.unpack(">I", raw_len)
    data = _recvall(sock, length)
    if data is None:
        raise ConnectionError("Server disconnected during payload recv")
    return pickle.loads(data)


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        packet = sock.recv(n - len(buf))
        if not packet:
            return None
        buf.extend(packet)
    return bytes(buf)


# ── Key mapping: ManiFeel env keys -> MMDF modality names ────────────────────
# Images: (B, T, C, H, W), normalised to [0, 1] float32
# Vectors: (B, T, D) float32
_KEY_MAP = {
    "right_tactile_camera_taxim": "taxim",
    "wrist": "wrist",
    "state": "state",
}
_IMAGE_MODALITIES = {"taxim", "wrist"}


def _prepare_obs(obs_dict: dict) -> dict:
    """
    Remap ManiFeel obs keys to MMDF modality names and convert to numpy.
    Images are normalised to [0, 1] float32 if not already.
    """
    out = {}
    for src, dst in _KEY_MAP.items():
        if src not in obs_dict:
            continue
        arr = obs_dict[src]
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = arr.astype(np.float32)
        if dst in _IMAGE_MODALITIES and arr.max() > 1.5:
            arr = arr / 255.0
        out[dst] = arr
    return out


# ── Socket policy ─────────────────────────────────────────────────────────────

class MMDFSocketPolicy(BaseImagePolicy):
    """
    Thin wrapper that forwards observations to the remote MMDF policy server
    and returns actions.

    Args:
        server_host: Hostname / IP of the policy server.
        server_port: TCP port the server is listening on.
        action_dim:  Expected action dimensionality (default 6 for plug task).
        connect_timeout: Seconds to keep retrying connection on startup.
    """

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 5556,
        action_dim: int = 6,
        connect_timeout: float = 120.0,
        n_obs_steps: int = 1,     # kept for compatibility; server manages history
        n_action_steps: int = 1,
    ):
        super().__init__()
        self.server_host = server_host
        self.server_port = server_port
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self._sock = None
        self._connect(connect_timeout)

    # ── Connection management ────────────────────────────────────────────────

    def _connect(self, timeout: float):
        deadline = time.time() + timeout
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.server_host, self.server_port))
                sock.settimeout(None)
                self._sock = sock
                print(f"[client] Connected to policy server at "
                      f"{self.server_host}:{self.server_port}")
                return
            except (ConnectionRefusedError, socket.timeout):
                if time.time() > deadline:
                    raise RuntimeError(
                        f"Could not connect to policy server at "
                        f"{self.server_host}:{self.server_port} "
                        f"within {timeout}s"
                    )
                print("[client] Waiting for policy server …", flush=True)
                time.sleep(2.0)

    # ── BaseImagePolicy interface ────────────────────────────────────────────

    def reset(self):
        """Tell the server to reset its observation history."""
        _send(self._sock, {"reset": True, "num_envs": 1})
        resp = _recv(self._sock)
        assert resp.get("status") == "reset_ok", f"Unexpected reset response: {resp}"

    def predict_action(self, obs_dict: dict) -> dict:
        """
        Args:
            obs_dict: Dict of torch tensors with ManiFeel observation keys.
                      Shapes: images (B, T, C, H, W), state (B, T, D).
        Returns:
            {"action": torch.Tensor of shape (B, n_action_steps, action_dim)}
        """
        obs_np = _prepare_obs(obs_dict)
        _send(self._sock, {"obs": obs_np, "reset": False})
        resp = _recv(self._sock)
        action_np = resp["action"]           # (B, n_action_steps, action_dim)
        action = torch.from_numpy(action_np)
        if action.ndim == 2:
            action = action.unsqueeze(1)     # legacy single-step -> (B, 1, D)
        return {"action": action}

    # ── Required by BaseImagePolicy (training stubs) ─────────────────────────

    def set_normalizer(self, normalizer):
        pass  # normalizer lives on the server side

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32
