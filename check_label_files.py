import pyarrow.parquet as pq

# ラベルファイルのパス
label_file_path = 'YOUR_DATA_SET_DIRECTORY\\Waymo\\training\\lidar_box\\15448466074775525292_2920_000_2940_000.parquet'

# ラベルデータの確認
def check_label_file_content(label_file_path):
    try:
        # pyarrowを使用してデータを読み込む
        table = pq.read_table(label_file_path)
        print(table)  # テーブルの内容を表示
    except Exception as e:
        print(f"Error reading {label_file_path}: {e}")

# ラベルデータの確認
check_label_file_content(label_file_path)