import h5py
import os

h5_path = 'data/fix_line/scene_007_Target_009_height_100_ang_240.h5'

def format_size(size_bytes):
    """自动选择合适的单位"""
    if size_bytes < 1024:
        return f"{size_bytes:.0f} Bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    else:
        return f"{size_bytes/(1024**2):.2f} MB"

def print_size(name, obj):
    if isinstance(obj, h5py.Dataset):
        # 获取实际占用的存储大小
        raw_size = obj.id.get_storage_size()
        size_str = format_size(raw_size)
        print(f"Dataset: {name:<25} | Size: {size_str:>12}")

if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        print(f"File: {h5_path}")
        print("-" * 50)
        f.visititems(print_size)
        print("-" * 50)
        
        total_size = os.path.getsize(h5_path)
        print(f"Total Disk Size: {format_size(total_size)}")
else:
    print(f"File not found: {h5_path}")