import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import GradScaler, autocast

class PointPillars(nn.Module):
    def __init__(self):
        super(PointPillars, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)  # チャンネル数を減らす
        self.bn1 = nn.BatchNorm2d(32)  # バッチ正規化を追加
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # チャンネル数を減らす
        self.bn2 = nn.BatchNorm2d(64)  # バッチ正規化を追加
        self.fc1 = nn.Linear(64 * 1000, 128)  # 入力サイズと出力サイズを減らす
        self.fc2 = nn.Linear(128, 6)  # 出力サイズを6に変更

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # バッチ正規化を適用
        x = torch.relu(self.bn2(self.conv2(x)))  # バッチ正規化を適用
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.all_point_clouds, self.all_labels = pickle.load(f)
        self.max_points = min(max(len(pc) for pc in self.all_point_clouds), 1000)  # max_pointsを減らす
        self.max_labels = max(len(lbl[1]) for lbl in self.all_labels)

    def __len__(self):
        return len(self.all_point_clouds)

    def __getitem__(self, idx):
        pc = self.all_point_clouds[idx]
        lbl = self.all_labels[idx][1]
        if len(pc) > self.max_points:
            pc = pc[:self.max_points]
        padded_pc = np.pad(pc, ((0, self.max_points - len(pc)), (0, 0)), mode='constant', constant_values=0).astype(np.float32)
        padded_lbl = np.pad(lbl, (0, self.max_labels - len(lbl)), mode='constant', constant_values=-1).astype(np.float32)
        padded_pc = padded_pc.transpose(1, 0).reshape(4, self.max_points, 1)
        return torch.tensor(padded_pc, dtype=torch.float32), torch.tensor(padded_lbl, dtype=torch.float32)

def create_data_loader(file_path, batch_size=1):
    dataset = PointCloudDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, data_loader, criterion, optimizer, num_epochs=25, save_path='model.pth', save_interval=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scaler = GradScaler('cuda')  # AMPのためのスケーラーを作成
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):  # AMPを使用
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")
    torch.save(model.state_dict(), save_path)
    print("Final model saved successfully.")

model_training = PointPillars()
criterion = nn.MSELoss()
optimizer_training = optim.Adam(model_training.parameters(), lr=0.0001)  # 学習率を下げる
data_loader_training = create_data_loader('YOUR_DATA_SET_DIRECTORY\\Waymo\\processed_data\\training\\processed_point_clouds.pkl', batch_size=1)
train_model(model_training, data_loader_training, criterion, optimizer_training, save_path='YOUR_DATA_SET_DIRECTORY\\Waymo\\processed_data\\pointpillars_model_training.pth')