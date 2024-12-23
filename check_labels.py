import pandas as pd
import pyarrow.parquet as pq

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

# ラベルデータの確認
def check_labels(label_file_path):
    df_labels = load_parquet_file(label_file_path)
    if df_labels is not None:
        print(df_labels.head())  # 最初の数行を表示
        print(df_labels.columns)  # カラム名を表示
        print(f"Number of rows: {len(df_labels)}")  # 行数を表示
    else:
        print(f"Failed to load label file: {label_file_path}")

# ラベルファイルのパス
label_file_path = 'YOUR_DATA_SET_DIRECTORY\\Waymo\\training\\lidar_box\\15448466074775525292_2920_000_2940_000.parquet'

# ラベルデータの確認
check_labels(label_file_path)