import pickle

def check_data(file_path):
    with open(file_path, 'rb') as f:
        all_point_clouds, all_labels = pickle.load(f)
    print(f"Number of point clouds: {len(all_point_clouds)}")
    print(f"Number of labels: {len(all_labels)}")
    if len(all_labels) > 0:
        print(f"First label: {all_labels[0]}")
        print(f"First point cloud: {all_point_clouds[0]}")

# データの確認
check_data('YOUR_DATA_SET_DIRECTORY\\processed_data\\training\\processed_point_clouds.pkl')