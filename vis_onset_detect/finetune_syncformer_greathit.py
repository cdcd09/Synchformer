import os, sys
from pathlib import Path

# -------- 경로 세팅 --------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, "/workspace")                    # utils 및 model 패키지 접근 가능하게
sys.path.insert(0, str(PROJECT_ROOT))

# -------- 이후 나머지 import 실행 --------
import glob, argparse, json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
import numpy as np
import wandb
from model.sync_model import Synchformer as SynchFormer


from model.sync_model import Synchformer as SynchFormer

# ------------------------------------------------------
# 1. Argument Parser
# ------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="/data/vis_jh/vis-data")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--pretrained_cfg", type=str, default="/workspace/configs/syncformer_audioset_pretrained.yaml")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="SynchFormer_GreatHit")
parser.add_argument("--wandb_name", type=str, default="greathit_finetune")
args = parser.parse_args()

# ------------------------------------------------------
# 2. 모델 로드 (YAML config 사용)
# ------------------------------------------------------
cfg = yaml.safe_load(open(args.pretrained_cfg, "r"))
model_cfg = cfg["model"]["params"]

# SynchFormer 인스턴스 생성
model = SynchFormer(**model_cfg)

# checkpoint 불러오기
ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
state_dict = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)

# fine-tune용 classifier 추가
model.head = nn.Linear(model.embed_dim, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 분산 학습 (DDP)
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

# ------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------
class GreatHitDataset(Dataset):
    def __init__(self, root=args.root, fps=25, split_list=None):
        self.root = root
        self.fps = fps
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        if split_list is None:
            self.samples = sorted(glob.glob(os.path.join(root, "*_mic.wav")))
        else:
            with open(split_list, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            self.samples = [os.path.join(root, f"{vid}_mic.wav") for vid in lines]

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        vid_id = os.path.basename(wav_path).replace("_mic.wav","")
        frame_dir = f"{self.root}/{vid_id}"
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frames", "*.jpg")))
        imgs = torch.stack([self.tf(Image.open(p)) for p in frame_paths])
        label_path = f"{self.root}/labels_framewise/{vid_id}_labels.npy"
        y = np.load(label_path)
        waveform, sr = torchaudio.load(wav_path)
        return imgs, waveform, torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

train_txt = os.path.join(args.root, "train.txt")
test_txt  = os.path.join(args.root, "test.txt")
train_set = GreatHitDataset(split_list=train_txt)
test_set  = GreatHitDataset(split_list=test_txt)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ------------------------------------------------------
# 4. Training 설정
# ------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

if args.use_wandb:
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    wandb.watch(model)

# ------------------------------------------------------
# 5. Training Loop
# ------------------------------------------------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for imgs, wav, label in train_loader:
        imgs, wav, label = imgs.to(device), wav.to(device), label.to(device)
        optimizer.zero_grad()
        out = model.forward_features(wav, imgs)  # (B,T,D)
        out = model.head(out.mean(2))  # 평균 pooling
        loss = criterion(out.squeeze(), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = 0, 0
        for imgs, wav, label in test_loader:
            imgs, wav, label = imgs.to(device), wav.to(device), label.to(device)
            out = model.forward_features(wav, imgs)
            out = model.head(out.mean(2))
            val_loss += criterion(out.squeeze(), label).item()
            preds = torch.sigmoid(out.squeeze()) > 0.5
            val_acc += (preds == (label>0.5)).float().mean().item()
        val_loss /= len(test_loader)
        val_acc  /= len(test_loader)

    print(f"[{epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if args.use_wandb:
        wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "val_acc": val_acc})

# ------------------------------------------------------
# 6. Save checkpoint
# ------------------------------------------------------
torch.save(model.state_dict(), "greathit_finetuned.pt")
if args.use_wandb:
    wandb.save("greathit_finetuned.pt")
    wandb.finish()
