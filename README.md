# CARLAシミュレータでLiDARを用いた物体検出（PointPillars法）

このリポジトリは、PointPillars法を使用してCARLAシミュレータでLiDARデータを処理し、物体検出を行うシステムの実装を含みます。

3D LiDAR点群データの物体検出のための深層学習モデルのトレーニングと評価に焦点を当てています。

## 概要
本システムは以下のワークフローで構成されています：

1. **データセット**: Waymo Open Datasetを主なLiDAR点群データのソースとして使用します。
2. **データ前処理**: PointPillars法に適した点群データを生成するために、LiDARデータを前処理します。
3. **モデル学習**: PyTorchを使用してPointPillarsネットワークをトレーニングし、点群内の物体を検出します。
4. **シミュレーション**: CARLAシミュレータを使用して、シミュレートされたLiDARデータを生成し、仮想環境でモデルをテストします。
5. **可視化**: OpenCVを用いて2Dバウンディングボックスを、Matplotlibを用いて3Dの結果を可視化します。

## 特徴
- **点群データの前処理**: 生のLiDARデータを効率的にグリッドベースのフォーマットに変換。
- **PointPillarsの実装**: 畳み込みニューラルネットワーク（CNN）を使用した物体検出。
- **シミュレート環境**: CARLAを用いた現実的な仮想シナリオでのテスト。
- **可視化**: 検出結果を解釈しやすい形式で生成。

## 必要要件
- Python 3.10.9
- PyTorch 2.4.1
- CUDA（cuDNNサポート付き）
- OpenCV 4.7
- CARLA Simulator v0.9.12
- Waymo Open Dataset v2.0.1

## インストール
1. このリポジトリをクローンします：
   ```bash
   git clone https://github.com/paraccoli/PointPillars-LiDAR.git
   cd PointPillars-LiDAR
   ```
2. 依存関係をインストールします：
   ```bash
   pip install -r requirements.txt
   ```
3. [公式ドキュメント](https://carla.readthedocs.io/)に従い、CARLAシミュレータをセットアップします。
4. Waymo Open Datasetをダウンロードし、点群データを前処理します。

## 使用方法

### 前処理
Waymo Open DatasetからLiDARデータを前処理するには：
```bash
python Testing_preprocessing.py
```

```bash
python Training_preprocessing.py
```

```bash
python validation_preprocessing.py
```

### トレーニング
PointPillarsモデルをトレーニングするには：
```bash
python train_pointpillars_training.py
```

### テストと可視化
CARLAシミュレータを起動し、仮想環境でトレーニング済みモデルをテストします：
```bash
python manual_control.py
```

## プロジェクト構成
- `train_pointpillars_training.py`: PointPillarsモデルのトレーニング用スクリプト。
- `manual_control.py`: CARLAシミュレータでマニュアル運転をしながらトレーニング済みモデルを実行するスクリプト。
- `requirements.txt`: 必要なPythonパッケージ。
- `README.md`: このファイル。

## 実験結果
- シミュレーションで車両の検出に成功したが精度が低い。
- 2Dバウンディングボックスと3Dプロットで検出結果を可視化。(難あり)

## 今後の展望
- PointRCNNへの移行による精度向上。
- センサーフュージョン技術の実装。
- マルチクラス検出とリアルタイム最適化の追加。

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。詳細はLICENSEファイルをご確認ください。

## 謝辞
- [Waymo Open Dataset](https://waymo.com/open/)
- [CARLA Simulator](https://carla.org/)
- [PointPillars研究論文](https://arxiv.org/abs/1812.05784)

---

# LiDAR-Based Object Detection in CARLA Simulator using PointPillars

This repository contains the implementation of a system that processes LiDAR data using the PointPillars method for object detection in the CARLA simulator.

The project focuses on training and evaluating a deep learning model for object detection in 3D LiDAR point cloud data.

## Overview
The system utilizes the following workflow:

1. **Dataset**: Waymo Open Dataset is used as the primary source of LiDAR point cloud data.
2. **Data Preprocessing**: LiDAR data is preprocessed to generate point cloud pillars suitable for the PointPillars method.
3. **Model Training**: The PointPillars network is trained using PyTorch to detect objects in the point cloud.
4. **Simulation**: CARLA simulator is used to generate simulated LiDAR data and test the trained model in a virtual environment.
5. **Visualization**: Detection results are visualized using OpenCV for 2D bounding boxes and Matplotlib for 3D visualization.

## Features
- **Point Cloud Preprocessing**: Efficiently converts raw LiDAR data into a grid-based format for deep learning.
- **PointPillars Implementation**: A convolutional neural network (CNN) for object detection.
- **Simulated Environment**: Test the system in a realistic virtual scenario using CARLA.
- **Visualization**: Generate interpretable results for object detection.

## Requirements
- Python 3.10.9
- PyTorch 2.4.1
- CUDA with cuDNN support
- OpenCV 4.7
- CARLA Simulator v0.9.12
- Waymo Open Dataset v2.0.1

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/paraccoli/PointPillars-LiDAR.git
   cd PointPillars-LiDAR
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the CARLA simulator by following the [official documentation](https://carla.readthedocs.io/).
4. Download the Waymo Open Dataset and preprocess the point cloud data.

## Usage

### Preprocessing
To preprocess LiDAR data from the Waymo Open Dataset:
```bash
python Testing_preprocessing.py
```

```bash
python Training_preprocessing.py
```

```bash
python validation_preprocessing.py
```

### Training
To train the PointPillars model:
```bash
python train_pointpillars_training.py
```

### Testing and Visualization
Run the CARLA simulator and test the trained model in the virtual environment:
```bash
python manual_control.py
```

## Project Structure
- `train_pointpillars_training.py`: Script for training the PointPillars model.
- `manual_control.py`: Script for manual driving in the CARLA simulator while testing the trained model.
- `requirements.txt`: Required Python packages.
- `README.md`: This file.

## Results
- Successfully detected vehicles in simulations, but with low accuracy.
- Visualized detection results with 2D bounding boxes and 3D plots (with limitations).

## Future Work
- Transition to PointRCNN for improved accuracy.
- Implement sensor fusion techniques.
- Add multi-class detection and real-time optimization.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Waymo Open Dataset](https://waymo.com/open/)
- [CARLA Simulator](https://carla.org/)
- [PointPillars Research Paper](https://arxiv.org/abs/1812.05784)
