import argparse
import os
import time

import pandas as pd
import torch
import torch.optim as optim

from libs.checkpoint import resume, save_checkpoint
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.graph import make_graphs
from libs.helper import evaluate, train
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.transformer import DataTransform

random_seed = 1234


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        train OpenPose for object detection with COCO Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    if os.path.exists(os.path.join(result_path, "final_model.prm")):
        print("Already done.")
        return

    device = get_device(allow_only_gpu=True)

    train_loader = get_dataloader(
        csv_file=config.train_csv,
        json_file=config.json_file,
        phase="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=DataTransform(),
    )

    val_loader = get_dataloader(
        csv_file=config.val_csv,
        json_file=config.json_file,
        phase="val",
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=DataTransform(),
    )

    model = get_model(pretrained=config.pretrained)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.0001,
    )

    begin_epoch = 0
    best_loss = float("inf")
    # TODO 評価指標の検討
    log = pd.DataFrame(
        columns=[
            "epoch",
            "lr",
            "train_time[sec]",
            "train_loss",
            "val_time[sec]",
            "val_loss",
        ]
    )

    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

        log_path = os.path.join(result_path, "log.csv")
        assert os.path.exists(log_path), "there is no checkpoint at the result folder"
        log = pd.read_csv(log_path)

    criterion = get_criterion()

    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        start = time.time()
        train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            interval_of_progress=10,
        )
        train_time = int(time.time() - start)

        start = time.time()
        val_loss = evaluate(
            val_loader,
            model,
            criterion,
            device,
        )
        val_time = int(time.time() - start)

        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(result_path, "best_model.prm"),
            )

        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        tmp = pd.Series(
            [
                epoch,
                optimizer.param_groups[0]["lr"],
                train_time,
                train_loss,
                val_time,
                val_loss,
            ],
            index=log.columns,
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"), index=False)
        make_graphs(os.path.join(result_path, "log.csv"))

        print(
            """epoch: {}\tepoch time[sec]: {}\tlr: {}\ttrain loss: {:.4f}\t\
            val loss: {:.4f}
            """.format(
                epoch,
                train_time + val_time,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
            )
        )

    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    os.remove(os.path.join(result_path, "checkpoint.pth"))

    print("Done")


if __name__ == "__main__":
    main()
