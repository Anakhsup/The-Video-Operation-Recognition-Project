import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import TimesformerForVideoClassification

# ==== Константы ====
MODEL_PATH = "model/weights.pth"
NUM_CLASSES = 9
FRAME_COUNT = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Список классов ====
CLASSES = [
    "measure", "pipe_work", "HRW_work", "pipe_vertical_movement", "cleaning",
    "gaskets", "spider_landing", "unscrewing PCP", "crane_vertical_movement"
]

# ==== Загрузка модели ====
model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

# ==== Dataset для одного видео ====
class InferenceVideoDataset(Dataset):
    def __init__(self, video_paths, frame_count=8):
        self.video_paths = video_paths
        self.frame_count = frame_count

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.frame_count)

        for i in range(self.frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        while len(frames) < self.frame_count:
            frames.append(frames[-1])

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
        frames = torch.tensor(frames, dtype=torch.float32) / 255.0

        return frames

# ==== Функция предсказания (с фиксированной моделью) ====
def predict_video(video_path):
    dataset = InferenceVideoDataset([video_path], frame_count=FRAME_COUNT)
    frames = dataset[0]

    with torch.no_grad():
        inputs = frames.unsqueeze(0).to(DEVICE)
        outputs = model(inputs).logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    return {
        "class": CLASSES[pred_class],
        "confidence": confidence,
    }
