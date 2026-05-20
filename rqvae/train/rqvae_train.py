import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import CustomNpzFile
from rqvae_model import RQVAE
from trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE on item embeddings.")

    parser.add_argument("--data_path", type=str, default="outputs/sasrec/cache/emb/emb/emb",
                        help="Directory that contains one or more .npz files with embs and ids.")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/rqvae/ckpt",
                        help="Root directory for RQ-VAE checkpoints.")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device or cpu")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")

    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8192, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=20, help="eval interval in epochs")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="constant or linear")
    parser.add_argument("--warmup_epochs", type=int, default=50, help="warmup epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 regularization weight")

    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=str2bool, default=False, help="use batch norm")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"], help="reconstruction loss")
    parser.add_argument("--kmeans_init", type=str2bool, default=True, help="initialize codebooks with kmeans")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iterations")
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0],
                        help="Sinkhorn epsilons, one value for each codebook")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iterations")

    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[2048, 2048, 1024],
                        help="codebook size for each residual quantizer")
    parser.add_argument("--e_dim", type=int, default=32, help="codebook embedding size")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0, help="quantization loss weight")
    parser.add_argument("--beta", type=float, default=0.25, help="commitment loss beta")
    parser.add_argument("--layers", type=int, nargs="+", default=[256, 128, 64],
                        help="encoder hidden sizes; decoder uses the reverse")
    parser.add_argument("--save_limit", type=int, default=100)
    parser.add_argument("--world_size", type=int, default=1, help="reserved for compatibility")

    return parser.parse_args()


def collect_npz_files(data_path):
    data_path = os.path.abspath(data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"RQ-VAE data_path is not a directory: {data_path}")

    files = [
        os.path.join(data_path, name)
        for name in sorted(os.listdir(data_path))
        if name.endswith(".npz")
    ]
    if not files:
        raise FileNotFoundError(f"No .npz files found in RQ-VAE data_path: {data_path}")
    return files


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, data_path_list):
    data = CustomNpzFile(data_path_list)
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        shuffle=True,
        drop_last=True,
    )

    if len(data_loader) == 0:
        raise RuntimeError(
            f"Data has {len(data)} rows, smaller than batch_size={args.batch_size} with drop_last=True."
        )

    device = torch.device(args.device)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)

    trainer = Trainer(args, model, len(data_loader), device)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(f"Best loss: {best_loss}")
    print(f"Best collision rate: {best_collision_rate}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    data_path_list = collect_npz_files(args.data_path)
    print(f"RQ-VAE training data path: {os.path.abspath(args.data_path)}")
    print(f"Found {len(data_path_list)} npz file(s).")
    print(f"RQ-VAE checkpoint root: {os.path.abspath(args.ckpt_dir)}")
    print(f"Codebook sizes: {args.num_emb_list}")

    train(args, data_path_list)
