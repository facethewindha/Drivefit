import os

box_info_dir = "./datasets/box_info"

total = 0

print("统计每个天气文件夹数量：\n")

for folder in os.listdir(box_info_dir):
    folder_path = os.path.join(box_info_dir, folder)

    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        count = len(files)

        print(f"{folder}: {count} 个文件")
        total += count

print("\n======================")
print(f"总文件数: {total}")
print("======================")
