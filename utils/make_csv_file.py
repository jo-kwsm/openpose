import argparse
import glob, json, os, sys
from typing import Dict, List

import pandas as pd

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="make csv files for COCO dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/",
        help="path to a dataset dirctory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="a directory where csv files will be saved",
    )

    return parser.parse_args()

def main() -> None:
    args = get_arguments()

    data: Dict[str, Dict[str, List[str]]] = {
        "train": {
            "image_path": [],
            "mask_path": [],
            "meta_data_idx": [],
        },
        "val": {
            "image_path": [],
            "mask_path": [],
            "meta_data_idx": [],
        },
        "test": {
            "image_path": [],
            "mask_path": [],
            "meta_data_idx": [],
        },
    }

    json_path = os.path.join(args.dataset_dir, "COCO.json")
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data_json = data_this["root"]
    n_sample = len(data_json)

    mask_path_template = os.path.join(args.dataset_dir, "mask/%s2014/mask_COCO_%s2014_%s.jpg")

    for idx in range(n_sample):
        if data_json[idx]["isValidation"] != 0.:
            stage_name = "val"
        else:
            stage_name = "train"

        id_name = data_json[idx]['img_paths'][-16:-4]
        image_path = os.path.join(args.dataset_dir, data_json[idx]['img_paths'])
        mask_path = mask_path_template%(stage_name, stage_name, id_name)
        data[stage_name]["image_path"].append(image_path)
        data[stage_name]["mask_path"].append(mask_path)
        data[stage_name]["meta_data_idx"].append(idx)

    # list を DataFrame に変換
    train_df = pd.DataFrame(
        data["train"],
        columns=["image_path", "mask_path", "meta_data_idx"],
    )

    val_df = pd.DataFrame(
        data["val"],
        columns=["image_path", "mask_path", "meta_data_idx"],
    )

    test_df = pd.DataFrame(
        data["test"],
        columns=["image_path", "mask_path", "meta_data_idx"],
    )

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, "train.csv"), index=None)
    val_df.to_csv(os.path.join(args.save_dir, "val.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test.csv"), index=None)

    print("Finished making csv files.")


if __name__ == "__main__":
    main()
