import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class PointRCNN(nn.Module):
    def __init__(self):
        super(PointRCNN, self).__init__()
        # モデルの定義をここに記述
        self.conv1 = nn.Conv1d(4, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 5)  # 5クラスの物体検出

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, 2)[0]  # グローバル最大プーリング
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(file_path):
    with open(file_path, 'rb') as f:
        all_point_clouds, all_labels = pickle.load(f)
    max_points = max(len(pc) for pc in all_point_clouds)
    padded_point_clouds = [np.pad(pc, ((0, max_points - len(pc)), (0, 0)), mode='constant', constant_values=0) for pc in all_point_clouds]
    padded_labels = [np.pad(lbl, (0, max_points - len(lbl)), mode='constant', constant_values=-1) for lbl in all_labels]
    return padded_point_clouds, padded_labels

def create_data_loader(padded_point_clouds, padded_labels, batch_size=16):
    dataset = TensorDataset(torch.tensor(padded_point_clouds, dtype=torch.float32).permute(0, 2, 1), torch.tensor(padded_labels, dtype=torch.long))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train_model(model, data_loader, criterion, optimizer, num_epochs=25, save_path='model.pth', save_interval=5):
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 5), labels.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        
        # 定期的にモデルを保存
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'{save_path}_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")

    # 最終的なモデルの保存
    torch.save(model.state_dict(), save_path)
    print("Final model saved successfully.")

# モデルのインスタンス化
model_training = PointRCNN()
criterion = nn.CrossEntropyLoss()
optimizer_training = optim.Adam(model_training.parameters(), lr=0.001)

# データの読み込み
padded_point_clouds_training, padded_labels_training = load_data('YOUR_DATA_SET_DIRECTORY\\Waymo\\processed_data\\training\\processed_point_clouds.pkl')

# データローダーの作成
data_loader_training = create_data_loader(padded_point_clouds_training, padded_labels_training, batch_size=16)

# トレーニングの実行
train_model(model_training, data_loader_training, criterion, optimizer_training, save_path='YOUR_DATA_SET_DIRECTORY\\Waymo\\processed_data\\pointrcnn_model_training.pth')