from tqdm import tqdm
import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser(description="Split Ithaca365 dataset into weather scenarios.")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to resized dataset folder, e.g. ./datasets/Ithaca365/Ithaca365-256")
parser.add_argument("--save_path", type=str, required=True,
                    help="Output folder, e.g. ./datasets/Ithaca365/Ithaca365-scenario")
parser.add_argument("--scene_json_path", type=str, required=True,
                    help="Path to scene.json, e.g. ./datasets/v2.21/scene.json")
parser.add_argument("--weather_json_path", type=str, required=True,
                    help="Path to weather.json, e.g. ./datasets/v2.21/weather.json")


def build_scene_to_weather(weather_json_data):
    """
    官方 schema: weather.json 每条记录是:
      {"token":..., "description":"night", "scenes":[scene_token1, scene_token2, ...]}
    所以我们构建:
      scene_token -> weather_description
    """
    scene2weather = {}
    for w in weather_json_data:
        desc = w.get("description", "unknown")
        for scene_token in w.get("scenes", []):
            # 如果一个 scene 同时出现在多个天气里（理论上不应该），后来的会覆盖前面的
            scene2weather[scene_token] = desc
    return scene2weather


def split2scenario(dataset_path, save_path, scene_json_path, weather_json_path):
    with open(scene_json_path, "r", encoding="utf-8") as fp:
        scene_json_data = json.load(fp)

    with open(weather_json_path, "r", encoding="utf-8") as fp:
        weather_json_data = json.load(fp)

    scene2weather = build_scene_to_weather(weather_json_data)

    os.makedirs(save_path, exist_ok=True)

    # 遍历每个 scene（对应你数据里的一天/一个日期文件夹）
    for scene in tqdm(scene_json_data, desc="Splitting by weather"):
        date_folder = scene["name"]          # e.g. "01-16-2022" -> 对应 dataset_path 下的文件夹
        scene_token = scene["token"]         # 用它去 weather.json 里查归属天气

        date_path = os.path.join(dataset_path, date_folder)
        if not os.path.isdir(date_path):
            raise FileNotFoundError(f"Date folder not found: {date_path}")

        weather_desc = scene2weather.get(scene_token, "unknown")

        # 输出目录：save_path/weather_desc/
        weather_path = os.path.join(save_path, weather_desc)
        os.makedirs(weather_path, exist_ok=True)

        # 把该日期文件夹下的所有图片复制到对应天气目录
        for img_file in os.listdir(date_path):
            src = os.path.join(date_path, img_file)
            if os.path.isfile(src):
                shutil.copy(src, weather_path)


if __name__ == "__main__":
    args = parser.parse_args()
    split2scenario(args.dataset_path, args.save_path, args.scene_json_path, args.weather_json_path)
