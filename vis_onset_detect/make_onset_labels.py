'''
vis dataset의 *time_stamp.txt 파일을 읽어서 onset label 파일을 생성하는 스크립트

_labels.npy 파일이 생성됨 frame 단위로 onset이 있는지 없는지 0/1로 표시


'''

# file: make_onset_labels.py
import os, glob, csv
import numpy as np

FPS = 25  # GreatHit 영상은 대부분 25fps
MAX_DURATION = 90  # 초 단위 (한 영상 길이 상한)

def read_times_file(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            t = float(parts[0])
            material, action, motion = parts[1:]
            labels.append((t, material, action, motion))
    return labels

def convert_to_frame_labels(labels, fps=FPS, max_t=MAX_DURATION):
    n_frames = int(max_t * fps)
    y = np.zeros(n_frames, dtype=int)
    for t, m, a, mo in labels:
        if a.lower() == "none":  # 소리 없는 구간
            continue
        frame_idx = int(t * fps)
        if frame_idx < n_frames:
            y[frame_idx] = 1  # onset 존재
    return y

if __name__ == "__main__":
    root = "/data/vis_jh/vis-data"
    out_dir = os.path.join(root, "labels_framewise")
    os.makedirs(out_dir, exist_ok=True)

    for txt_path in sorted(glob.glob(os.path.join(root, "*_times.txt"))):
        vid_id = os.path.basename(txt_path).replace("_times.txt", "")
        labels = read_times_file(txt_path)
        frame_labels = convert_to_frame_labels(labels)
        np.save(os.path.join(out_dir, f"{vid_id}_labels.npy"), frame_labels)
        print(f"[✓] {vid_id} → {frame_labels.sum()} events")
