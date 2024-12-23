import pandas as pd
import os
from tqdm import tqdm
import pyarrow.parquet as pq
import numpy as np
import json
import pickle

# データセットのディレクトリ
data_dir = "YOUR_DATA_SET_DIRECTORY\\Waymo"
output_dir = "YOUR_DATA_SET_DIRECTORY\\Waymo\\processed_data"

# データの読み込み関数
def load_parquet_file(file_path):
    try:
        # pyarrowを使用してデータを読み込む
        table = pq.read_table(file_path)
        df = table.to_pandas()
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ポイントクラウドデータの前処理関数
def preprocess_point_cloud(df_lidar, df_labels):
    point_clouds = []
    labels = []
    # LiDARデータの前処理
    range_image_strs = df_lidar['[LiDARComponent].range_image_return1.values'].values
    for range_image_str in range_image_strs:
        if isinstance(range_image_str, str):
            range_image = np.array(json.loads(range_image_str))
        else:
            range_image = np.array(range_image_str)
        if range_image.ndim == 1:
            range_image = range_image.reshape(-1, 4)
        valid_points = range_image[(range_image[:, 0] != -1) & (range_image[:, 3] != -1)]
        if valid_points.ndim == 2 and valid_points.shape[1] >= 4:
            x = valid_points[:, 0]
            y = valid_points[:, 1]
            z = valid_points[:, 2]
            intensity = valid_points[:, 3]
            points = np.vstack((x, y, z, intensity)).T
            point_clouds.append(points)
        else:
            print("Invalid range_image shape")
    
    # ラベルデータの前処理
    print(df_labels.columns)  # カラム名を表示して確認
    for _, row in df_labels.iterrows():
        # 適切なカラム名を使用
        label = row['[LiDARBoxComponent].type']  # 'type'カラムを使用
        box = row[['[LiDARBoxComponent].box.center.x', '[LiDARBoxComponent].box.center.y', '[LiDARBoxComponent].box.center.z', '[LiDARBoxComponent].box.size.x', '[LiDARBoxComponent].box.size.y', '[LiDARBoxComponent].box.size.z']].values
        labels.append((label, box))
    
    return point_clouds, labels

# 各ファイルの前処理を行う関数
def process_all_files(lidar_folder, label_folder, output_folder):
    all_point_clouds = []
    all_labels = []
    for root, _, files in os.walk(lidar_folder):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.parquet'):
                lidar_file_path = os.path.join(root, file)
                label_file_path = os.path.join(label_folder, file)
                df_lidar = load_parquet_file(lidar_file_path)
                df_labels = load_parquet_file(label_file_path)
                if df_lidar is not None and df_labels is not None:
                    print(f"Loaded files: {lidar_file_path}, {label_file_path}")
                    point_clouds, labels = preprocess_point_cloud(df_lidar, df_labels)
                    print(f"Processed {len(point_clouds)} point clouds")
                    if point_clouds:
                        all_point_clouds.extend(point_clouds)
                        all_labels.extend(labels)
                else:
                    print(f"Failed to load files: {lidar_file_path}, {label_file_path}")
    # 前処理されたデータを保存
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'processed_point_clouds.pkl'), 'wb') as f:
        pickle.dump((all_point_clouds, all_labels), f)
    
    # ポイントクラウドの数とラベルの数を表示
    print(f"Total number of point clouds: {len(all_point_clouds)}")
    print(f"Total number of labels: {len(all_labels)}")

# 各ディレクトリのパス
directory = "training"
lidar_folder = os.path.join(data_dir, directory, "lidar")
label_folder = os.path.join(data_dir, directory, "lidar_box")
output_folder = os.path.join(output_dir, directory)

# 前処理を実行
if os.path.exists(lidar_folder) and os.path.exists(label_folder):
    process_all_files(lidar_folder, label_folder, output_folder)
else:
    print(f"Folder does not exist: {lidar_folder} or {label_folder}")