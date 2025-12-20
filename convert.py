import os
import json
import h5py
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
import pandas as pd
import tempfile
import subprocess
import traceback
import os


# ========================= 配置区 =========================
# repo_id = "hhuangbt/Folding_50_12_13"

original_dir = "/home/agilex/cobot_magic/collect_data/data/folding_clothes_251/"
out_dir = "/home/agilex/cobot_magic/collect_data/data/folding_clothes_251_lerobot/"

TEST_ONLY_FIRST_N = None  # 设为 None 运行全部


VIDEO_CODEC = 'libx264'  
# VIDEO_CODEC = 'libaom-av1' 
# VIDEO_CODEC = 'libsvtav1'

# # ========================= 下载与准备 =========================
# print("正在下载数据集...")
# snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=original_dir , ignore_patterns=["*.mp4"])
# print("下载完成。")



if not os.path.exists(original_dir):
    raise FileNotFoundError(f"未找到 train 目录: {original_dir}")

hdf5_files = [f for f in os.listdir(original_dir) if f.endswith('.hdf5')]
hdf5_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
print(f"找到 {len(hdf5_files)} 个 HDF5 文件。")

if TEST_ONLY_FIRST_N is not None:
    hdf5_files = hdf5_files[:TEST_ONLY_FIRST_N]

data_dir = os.path.join(out_dir, "data")
videos_dir = os.path.join(out_dir, "videos")
meta_dir = os.path.join(out_dir, "meta")
episodes_dir = os.path.join(meta_dir, "episodes")
temp_dir = tempfile.mkdtemp(prefix="lerobot_temp_")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)
os.makedirs(episodes_dir, exist_ok=True)

cameras = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
fps = 20
chunks_size = 1000
data_files_size_in_mb = 100
video_files_size_in_mb = 500
rows_per_file = int(data_files_size_in_mb * 1024 * 1024 / 500)

# ========================= 统计收集（用于完整 stats） =========================
from scipy.stats import scoreatpercentile

all_timestamps = []
all_time_stamps = []
all_frame_indices = []
all_episode_indices = []
all_indices = []
all_task_indices = []

all_states = []
all_actions = []

# 图像像素采样（每 episode 采样最多 10 帧）
sampled_images = {cam: [] for cam in cameras}  # list of (H, W, 3) float32 arrays

global_time_offset = 0.0

# ========================= 状态变量 =========================
episode_index = 0
global_index = 0
task_index = 0

current_data_rows = []
data_chunk_idx = 0
data_file_idx = 0

video_chunk_idx = {cam: 0 for cam in cameras}
video_file_idx = {cam: 0 for cam in cameras}
current_temp_videos = {cam: [] for cam in cameras}
current_video_size = {cam: 0 for cam in cameras}
current_frame_offset = {cam: 0 for cam in cameras}

episode_meta_rows = []

# ========================= 函数 =========================
def flush_data():
    global data_chunk_idx, data_file_idx
    if not current_data_rows:
        return
    print(f"   → 写入数据文件: chunk-{data_chunk_idx:03d}/file-{data_file_idx:03d}.parquet ({len(current_data_rows)} 行)")
    df = pd.DataFrame(current_data_rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    chunk_dir = os.path.join(data_dir, f"chunk-{data_chunk_idx:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    parquet_path = os.path.join(chunk_dir, f"file-{data_file_idx:03d}.parquet")
    pq.write_table(table, parquet_path)
    current_data_rows.clear()
    data_file_idx += 1
    if data_file_idx % chunks_size == 0:
        data_chunk_idx += 1
        data_file_idx = 0

def flush_video(cam):
    global video_chunk_idx, video_file_idx
    if not current_temp_videos[cam]:
        return
    cam_dir = os.path.join(videos_dir, cam, f"chunk-{video_chunk_idx[cam]:03d}")
    os.makedirs(cam_dir, exist_ok=True)
    video_path = os.path.join(cam_dir, f"file-{video_file_idx[cam]:03d}.mp4")
    print(f"   → 合并视频 {cam}: {video_path}")

    concat_list_path = os.path.join(temp_dir, f"concat_{cam}.txt")
    with open(concat_list_path, 'w') as f:
        for p in current_temp_videos[cam]:
            f.write(f"file '{p}'\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
                    "-c", "copy", video_path, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in current_temp_videos[cam]:
        os.remove(p)
    current_temp_videos[cam].clear()
    current_video_size[cam] = 0
    video_file_idx[cam] += 1
    if video_file_idx[cam] % chunks_size == 0:
        video_chunk_idx[cam] += 1
        video_file_idx[cam] = 0

# ========================= 主循环 =========================
print("\n开始处理 episodes...\n")

for hdf5_file in tqdm(hdf5_files, desc="总进度"):
    episode_path = os.path.join(original_dir, hdf5_file)
    print(f"\n[{episode_index}] 处理文件: {hdf5_file}")

    success = True
    episode_frames = 0
    episode_data_from = global_index

    try:
        with h5py.File(episode_path, 'r') as f:
            action = f['action'][()]
            obs = f['observations']
            qpos = obs['qpos'][()]
            qvel = obs['qvel'][()]
            effort = obs['effort'][()]

            n = action.shape[0]
            print(f"   → 帧数: {n}")

            time_stamp = f.get('time_stamp', np.arange(n) * (1.0 / fps))
            if hasattr(time_stamp, '__getitem__'):
                time_stamp = time_stamp[()]
            episode_start_time = time_stamp[0] if n > 0 else 0.0

            # 缺失字段用零填充
            eef = np.zeros((n, 14))
            eef_quaternion = np.zeros((n, 16))
            eef_6d = np.zeros((n, 20))
            eef_left_time = np.zeros((n, 1))
            eef_right_time = np.zeros((n, 1))
            base_left = np.zeros((n, 1))
            base_right = np.zeros((n, 1))

            for frame_idx in range(n):
                state = np.concatenate([
                    eef[frame_idx],
                    eef_quaternion[frame_idx],
                    eef_6d[frame_idx],
                    eef_left_time[frame_idx],
                    eef_right_time[frame_idx],
                    qpos[frame_idx],
                    qvel[frame_idx],
                    effort[frame_idx],
                    base_left[frame_idx],
                    base_right[frame_idx]
                ], dtype=np.float32)

                row = {
                    "observation.state": state.tolist(),
                    "action": action[frame_idx].tolist(),
                    "timestamp": [float(time_stamp[frame_idx] - episode_start_time)],
                    "time_stamp": [float(time_stamp[frame_idx] + global_time_offset)],
                    "frame_index": [frame_idx],
                    "episode_index": [episode_index],
                    "index": [global_index],
                    "task_index": [task_index]
                }
                current_data_rows.append(row)

                # 统计收集
                all_timestamps.append(float(time_stamp[frame_idx] - episode_start_time))
                all_time_stamps.append(float(time_stamp[frame_idx] + global_time_offset))
                all_frame_indices.append(frame_idx)
                all_episode_indices.append(episode_index)
                all_indices.append(global_index)
                all_task_indices.append(task_index)
                all_states.append(state.copy())
                all_actions.append(action[frame_idx].copy())

                global_index += 1
                episode_frames += 1

            # 更新全局时间偏移
            if n > 1:
                global_time_offset += (time_stamp[-1] - time_stamp[0])

            episode_video_info = {
                "episode_index": episode_index,
                "data_from": episode_data_from,
                "data_to": global_index - 1,
            }

            for cam in cameras:
                cam_key = f'observations/images/{cam}'
                if cam_key not in f:
                    print(f"   → 警告: {cam} 无图像数据")
                    continue

                images = f[cam_key][()]  # (n, 480, 640, 3) uint8
                print(f"   → {cam}: 加载原始像素 {images.shape}")

                frames = []
                for i in range(images.shape[0]):
                    img = images[i]
                    # 如果颜色不对，改成下面这行
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(img)

                # 采样用于图像统计（最多10帧）
                if len(frames) > 0:
                    sample_idx = np.linspace(0, len(frames)-1, num=min(10, len(frames)), dtype=int)
                    for idx in sample_idx:
                        sampled_images[cam].append(frames[idx].astype(np.float32) / 255.0)

                temp_video_path = os.path.join(temp_dir, f"ep{episode_index}_{cam}.mp4")
                clip = ImageSequenceClip(frames, fps=fps)
                clip.write_videofile(temp_video_path, codec=VIDEO_CODEC, audio=False, logger=None, ffmpeg_params=['-strict', '-2','-pix_fmt', 'yuv420p'])
                # clip.write_videofile(temp_video_path, codec=VIDEO_CODEC, audio=False, logger=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])
                # clip.write_videofile(temp_video_path, codec=VIDEO_CODEC, audio=False, logger=None, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"])
                
                temp_size = os.path.getsize(temp_video_path) / (1024**2)
                print(f"   → {cam} 视频生成成功，大小: {temp_size:.1f} MB")

                if current_video_size[cam] + temp_size > video_files_size_in_mb and current_temp_videos[cam]:
                    flush_video(cam)

                current_temp_videos[cam].append(temp_video_path)
                current_video_size[cam] += temp_size

                from_frame = current_frame_offset[cam]
                current_frame_offset[cam] += n
                to_frame = current_frame_offset[cam] - 1

                episode_video_info[f"video_{cam}_chunk"] = video_chunk_idx[cam]
                episode_video_info[f"video_{cam}_file"] = video_file_idx[cam]
                episode_video_info[f"video_{cam}_from_frame"] = from_frame
                episode_video_info[f"video_{cam}_to_frame"] = to_frame

            episode_meta_rows.append(episode_video_info)

            if len(current_data_rows) >= rows_per_file:
                flush_data()

    except Exception as e:
        print(f"!!! 处理失败: {e}")
        traceback.print_exc()
        success = False

    print(f"   → episode {episode_index} {'成功' if success else '部分失败'} ({episode_frames} 帧)")
    episode_index += 1

# ========================= 收尾 =========================
print("\n写入剩余数据...")
flush_data()
for cam in cameras:
    if current_temp_videos[cam]:
        flush_video(cam)

print(f"总 episode: {episode_index}，总帧: {global_index}")

# ========================= 完整 stats.json（像官方一样） =========================
print("\n生成完整 stats.json（含图像分位数）...")

stats = {}

def add_stats(name, values, is_image=False):
    if len(values) == 0:
        return
    arr = np.stack(values) if not is_image else np.concatenate([v.reshape(-1, 3) for v in values], axis=0)
    percs = [1, 10, 50, 90, 99]
    q = np.percentile(arr, percs, axis=0)

    if is_image:
        # per-channel nested list
        stats[name] = {
            "min": [[[float(arr[:, c].min())]] for c in range(3)],
            "max": [[[float(arr[:, c].max())]] for c in range(3)],
            "mean": [[[float(arr[:, c].mean())]] for c in range(3)],
            "std": [[[float(arr[:, c].std())]] for c in range(3)],
            "count": [len(values)],
            "q01": [[[float(q[0, c])]] for c in range(3)],
            "q10": [[[float(q[1, c])]] for c in range(3)],
            "q50": [[[float(q[2, c])]] for c in range(3)],
            "q90": [[[float(q[3, c])]] for c in range(3)],
            "q99": [[[float(q[4, c])]] for c in range(3)]
        }
    else:
        stats[name] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(len(values))],
            "q01": q[0].tolist(),
            "q10": q[1].tolist(),
            "q50": q[2].tolist(),
            "q90": q[3].tolist(),
            "q99": q[4].tolist()
        }

def add_scalar_stats(name, values):
    if len(values) == 0:
        return
    arr = np.array(values)
    percs = [1, 10, 50, 90, 99]
    q = np.percentile(arr, percs)
    stats[name] = {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(len(values))],
        "q01": [float(q[0])],
        "q10": [float(q[1])],
        "q50": [float(q[2])],
        "q90": [float(q[3])],
        "q99": [float(q[4])]
    }

add_scalar_stats("timestamp", all_timestamps)
add_scalar_stats("time_stamp", all_time_stamps)
add_scalar_stats("frame_index", all_frame_indices)
add_scalar_stats("episode_index", all_episode_indices)
add_scalar_stats("index", all_indices)
add_scalar_stats("task_index", all_task_indices)

add_stats("observation.state", all_states)
add_stats("action", all_actions)

for cam in cameras:
    add_stats(f"observation.images.{cam}", sampled_images[cam], is_image=True)

with open(os.path.join(meta_dir, "stats.json"), 'w') as f:
    json.dump(stats, f, indent=4)

# ========================= info.json（完全匹配官方格式） =========================
video_info = {
    "video.height": 480,
    "video.width": 640,
    "video.codec": "av1" if VIDEO_CODEC == 'libaom-av1' else "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "video.fps": 20,
    "video.channels": 3,
    "has_audio": False
}

info = {
    "codebase_version": "v3.0",
    "robot_type": "franka",
    "total_episodes": episode_index,
    "total_frames": global_index,
    "total_tasks": 1,
    "chunks_size": chunks_size,
    "data_files_size_in_mb": data_files_size_in_mb,
    "video_files_size_in_mb": video_files_size_in_mb,
    "fps": fps,
    "splits": {"train": f"0:{episode_index}"},
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
            "info": video_info
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
            "info": video_info
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
            "info": video_info
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [96],
            "names": [
                "eef_euler_0", "eef_euler_1", "eef_euler_2", "eef_euler_3", "eef_euler_4", "eef_euler_5", "eef_euler_6",
                "eef_euler_7", "eef_euler_8", "eef_euler_9", "eef_euler_10", "eef_euler_11", "eef_euler_12", "eef_euler_13",
                "eef_quat_0", "eef_quat_1", "eef_quat_2", "eef_quat_3", "eef_quat_4", "eef_quat_5", "eef_quat_6",
                "eef_quat_7", "eef_quat_8", "eef_quat_9", "eef_quat_10", "eef_quat_11", "eef_quat_12", "eef_quat_13", "eef_quat_14", "eef_quat_15",
                "eef6d_0", "eef6d_1", "eef6d_2", "eef6d_3", "eef6d_4", "eef6d_5", "eef6d_6", "eef6d_7", "eef6d_8",
                "eef6d_9", "eef6d_10", "eef6d_11", "eef6d_12", "eef6d_13", "eef6d_14", "eef6d_15", "eef6d_16", "eef6d_17", "eef6d_18", "eef6d_19",
                "eef_left_time", "eef_right_time",
                "qpos_0", "qpos_1", "qpos_2", "qpos_3", "qpos_4", "qpos_5", "qpos_6", "qpos_7", "qpos_8", "qpos_9", "qpos_10", "qpos_11", "qpos_12", "qpos_13",
                "qvel_0", "qvel_1", "qvel_2", "qvel_3", "qvel_4", "qvel_5", "qvel_6", "qvel_7", "qvel_8", "qvel_9", "qvel_10", "qvel_11", "qvel_12", "qvel_13",
                "effort_0", "effort_1", "effort_2", "effort_3", "effort_4", "effort_5", "effort_6", "effort_7", "effort_8", "effort_9", "effort_10", "effort_11", "effort_12", "effort_13",
                "qpos_left_time", "qpos_right_time"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": {
                "motors": [f"joint_action_{i}" for i in range(14)]
            }
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "time_stamp": {"dtype": "float32", "shape": [1], "names": {"values": ["global_timestamp"]}},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None}
    }
}
with open(os.path.join(meta_dir, "info.json"), 'w') as f:
    json.dump(info, f, indent=4)

# ========================= 其他 meta =========================
episode_df = pd.DataFrame(episode_meta_rows)
episode_table = pa.Table.from_pandas(episode_df, preserve_index=False)
os.makedirs(os.path.join(episodes_dir, "chunk-000"), exist_ok=True)
pq.write_table(episode_table, os.path.join(episodes_dir, "chunk-000", "file-000.parquet"))

tasks_data = [{"task_index": 0, "task_name": "folding_50_12_13"}]
pq.write_table(pa.Table.from_pandas(pd.DataFrame(tasks_data)), os.path.join(meta_dir, "tasks.parquet"))

print(f"\n转换完成！输出目录: {out_dir}")
print(f"临时目录（可删除）: {temp_dir}")