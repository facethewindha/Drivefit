import argparse
import torch


WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embed_ckpt", type=str, default="./pretrained_models/clip_similarity_embed.pt")
    p.add_argument("--source_weather", type=str, default="sunny")
    p.add_argument("--target_weather", type=str, default="rain")
    args = p.parse_args()

    ckpt = torch.load(args.embed_ckpt, map_location="cpu")
    key = "y_embedder.scenario_embedding_table.weight"
    if key not in ckpt:
        raise KeyError(f"Missing key: {key}")

    w = ckpt[key]
    src_idx = WEATHER_TO_IDX[args.source_weather]
    tgt_idx = WEATHER_TO_IDX[args.target_weather]
    null_idx = w.shape[0] - 1

    print("checkpoint:", args.embed_ckpt)
    print("weight shape:", tuple(w.shape))
    print("weather_to_idx:", WEATHER_TO_IDX)
    print(f"selected rows -> src={src_idx}({args.source_weather}), tgt={tgt_idx}({args.target_weather}), null={null_idx}")
    print("row norms:",
          float(w[src_idx].norm()),
          float(w[tgt_idx].norm()),
          float(w[null_idx].norm()))


if __name__ == "__main__":
    main()

