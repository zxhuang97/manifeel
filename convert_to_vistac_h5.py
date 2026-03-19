"""
Convert manifeel zarr datasets to vistac H5 format.

Zarr source format (manifeel):
  data/
    action        (N, 6 or 7)  float32
    state         (N, 7)       float32
    wrist         (N, H, W, 3) float32 [0,1]
    right_tactile_camera_taxim (N, 320, 240, 3) float32 [0,1]
    front/side/wrist_2/left_tactile_camera_taxim/...
    tactile_force_field_right  (N, 10, 14, 3)
    tactile_depth_right        (N, 10, 14)
  meta/
    episode_ends  (num_episodes,)  int64  (cumulative)

Output H5 format (vistac.py expects <ds_path>/<split>.h5):
  action           (N, D)
  state            (N, 7)
  wrist            vlen uint8     JPEG-encoded frames resized to 224×224 (one entry per step)
  taxim            vlen uint8     JPEG-encoded frames resized to 224×224 (one entry per step)
  trial_success    (num_episodes,) float32  (1.0 = success)
  episode_lens     (num_episodes,)
  episode_starts   (num_episodes,)
  episode_ends     (num_episodes,)
  [optional other modalities]

Output directory structure:
  <out_dir>/<task>/training.h5
  <out_dir>/<task>/validation.h5
  <out_dir>/<task>/debug.h5
"""

import argparse
import os
import sys
import numpy as np
import zarr
import h5py
import cv2
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


ZARR_TASKS = [
    "plug_quan_Aug02",
    "usb_quan_Aug05",
    "bulb_quan_Sep19",
    "blindinsert_quan_Aug15",
    "explore_quan_June17",
    "gear_quan_Sep15",
    "nutbolt_quan_July1",
    "pih_quan_June06",
    "sorting_quan_Aug8",
]

# Keys to copy verbatim (float arrays, lzf compression)
PLAIN_KEYS = ["action", "state", "tactile_force_field_right", "tactile_depth_right",
              "left_tactile_camera_taxim"]

# Camera image keys to JPEG-encode as vlen datasets; 'wrist' is always treated this way
JPEG_CAMERA_KEYS = ["wrist", "front", "side", "wrist_2"]


IMAGE_SIZE = (224, 224)


def encode_jpeg_vlen(images_nhwc, quality=95):
    """Encode (N, H, W, 3) float32 [0,1] array to list of JPEG byte arrays, resized to IMAGE_SIZE."""
    imgs_uint8 = (np.clip(images_nhwc, 0, 1) * 255).astype(np.uint8)
    encoded = []
    for img in imgs_uint8:
        if img.shape[:2] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        # zarr taxim images are RGB; cv2 expects BGR for imencode
        # img_bgr = img[..., ::-1]
        success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert success
        encoded.append(np.frombuffer(buf.tobytes(), dtype=np.uint8))
    return encoded


def convert_zarr_to_h5(zarr_path, out_h5_path, episode_indices, keys_to_save, taxim_key, jpeg_quality=90):
    """Convert a subset of episodes from a zarr store to a single H5 file."""
    store = zarr.open(zarr_path, mode='r')
    ep_ends = np.array(store['meta']['episode_ends'])
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    ep_lens = ep_ends - ep_starts

    # Gather step indices for selected episodes
    step_mask = np.zeros(int(ep_ends[-1]), dtype=bool)
    for i in episode_indices:
        step_mask[ep_starts[i]:ep_ends[i]] = True

    sel_ep_lens = ep_lens[episode_indices]
    sel_ep_starts = np.concatenate([[0], np.cumsum(sel_ep_lens)[:-1]])
    sel_ep_ends = np.cumsum(sel_ep_lens)
    num_sel_steps = int(sel_ep_lens.sum())

    # These keys are always JPEG-encoded; skip if accidentally included in keys_to_save
    JPEG_ONLY = set(JPEG_CAMERA_KEYS)

    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    with h5py.File(out_h5_path, 'w') as hf:
        # --- plain float arrays ---
        for key in keys_to_save:
            if key not in store['data'] or key in JPEG_ONLY:
                continue
            arr = store['data'][key]
            data = np.array(arr)[step_mask]
            hf.create_dataset(key, data=data, compression='lzf')
            print(f"  Saved {key}: {data.shape}")

        # --- taxim: JPEG-encoded variable-length ---
        if taxim_key and taxim_key in store['data']:
            print(f"  Encoding taxim ({taxim_key}) as JPEG vlen...")
            taxim_arr = store['data'][taxim_key]
            taxim_sel = np.array(taxim_arr)[step_mask]  # (N, H, W, 3) float32
            encoded = encode_jpeg_vlen(taxim_sel, quality=jpeg_quality)
            dt = h5py.special_dtype(vlen=np.uint8)
            dset = hf.create_dataset('taxim', (num_sel_steps,), dtype=dt)
            for i, enc in enumerate(encoded):
                dset[i] = enc
            print(f"  Saved taxim: {num_sel_steps} JPEG frames")

        # --- camera images (wrist, front, side, wrist_2): JPEG-encoded variable-length ---
        for cam_key in JPEG_CAMERA_KEYS:
            if cam_key not in store['data']:
                continue
            print(f"  Encoding {cam_key} as JPEG vlen...")
            cam_arr = store['data'][cam_key]
            cam_sel = np.array(cam_arr)[step_mask]  # (N, H, W, 3) float32
            encoded = encode_jpeg_vlen(cam_sel, quality=jpeg_quality)
            dt = h5py.special_dtype(vlen=np.uint8)
            dset = hf.create_dataset(cam_key, (num_sel_steps,), dtype=dt)
            for i, enc in enumerate(encoded):
                dset[i] = enc
            print(f"  Saved {cam_key}: {num_sel_steps} JPEG frames")

        # --- episode metadata ---
        trial_success = np.ones(len(episode_indices), dtype=np.float32)
        hf.create_dataset('trial_success', data=trial_success, compression='lzf')
        hf.create_dataset('episode_lens', data=sel_ep_lens.astype(np.int64), compression='lzf')
        hf.create_dataset('episode_starts', data=sel_ep_starts.astype(np.int64), compression='lzf')
        hf.create_dataset('episode_ends', data=sel_ep_ends.astype(np.int64), compression='lzf')
        print(f"  Saved metadata: {len(episode_indices)} episodes, {num_sel_steps} steps")


def convert_task(task, data_dir, out_dir, out_name, val_ratio, keys, taxim_key, jpeg_quality):
    """Convert one zarr task folder to H5 splits. Designed to run in a subprocess."""
    zarr_path = os.path.join(data_dir, task)
    if not os.path.exists(zarr_path):
        print(f"[SKIP] {task}: not found at {zarr_path}", flush=True)
        return

    store = zarr.open(zarr_path, mode='r')
    ep_ends = np.array(store['meta']['episode_ends'])
    num_eps = len(ep_ends)

    all_idx = np.arange(num_eps)
    n_val = max(1, int(num_eps * val_ratio))
    train_idx = all_idx[:-n_val]
    val_idx = all_idx[-n_val:]

    print(f"\n=== {task}: {num_eps} episodes -> {len(train_idx)} train / {n_val} val+debug ===", flush=True)

    splits = [
        ('training',   train_idx),
        ('validation', val_idx),
        ('debug',      val_idx),
    ]
    for split, ep_idx in splits:
        out_path = os.path.join(out_dir, out_name, f"{split}.h5")
        print(f"  [{task}:{split}] -> {out_path}", flush=True)
        convert_zarr_to_h5(zarr_path, out_path, ep_idx, keys, taxim_key, jpeg_quality)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/zixuanh/force_tool_tactile/manifeel/manifeel/data',
                        help='Root directory containing zarr task folders')
    parser.add_argument('--out_dir', type=str,
                        default='/home/zixuanh/force_tool_tactile/raw_datasets',
                        help='Output root directory; H5s go to <out_dir>/<task>/{training,validation,debug}/')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Task names to convert (default: all)')
    parser.add_argument('--name', type=str, default=None,
                        help='Output dataset name; defaults to the task name (only meaningful with a single --tasks entry)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of episodes (tail) used for validation and debug splits')
    parser.add_argument('--keys', nargs='+',
                        default=['action', 'state', 'tactile_force_field_right', 'tactile_depth_right'],
                        help='Float array keys to copy verbatim (wrist/camera keys are always JPEG-encoded separately)')
    parser.add_argument('--taxim_key', type=str, default='right_tactile_camera_taxim',
                        help='Key for tactile camera image to JPEG-encode as taxim (empty string to skip)')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help='JPEG quality for taxim encoding')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel worker processes (default: number of tasks)')
    args = parser.parse_args()

    tasks = args.tasks or ZARR_TASKS
    taxim_key = args.taxim_key if args.taxim_key else None

    if len(tasks) == 1:
        out_name = args.name if args.name else tasks[0]
        convert_task(tasks[0], args.data_dir, args.out_dir, out_name,
                     args.val_ratio, args.keys, taxim_key, args.jpeg_quality)
    else:
        n_workers = args.workers or len(tasks)
        print(f"Converting {len(tasks)} tasks with {n_workers} parallel workers...")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for task in tasks:
                out_name = task  # --name is only meaningful for single-task runs
                fut = pool.submit(convert_task, task, args.data_dir, args.out_dir, out_name,
                                  args.val_ratio, args.keys, taxim_key, args.jpeg_quality)
                futures[fut] = task
            for fut in as_completed(futures):
                task = futures[fut]
                exc = fut.exception()
                if exc:
                    print(f"[ERROR] {task}: {exc}", flush=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
