from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from dataset import *
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math
import warnings
import model

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):

    def lr_lambda(step):
        # warmup phase
        if step < warmup_steps:
            return step / float(warmup_steps)

        # cosine decay phase
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def lr_scheduler(optimizer, warmup_steps, total_steps):
    warnings.filterwarnings("ignore", message=".*epoch parameter in `scheduler.step\\(\\)` was not necessary.*")

    if total_steps <= warmup_steps:
        scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=total_steps)

    else:
        # Scheduler: ensure T_max >= 1
        cosine_T = max(1, total_steps - warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=1e-6)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    return scheduler

#cli
def parse_inputs():
    parser = ArgumentParser()

    #add arguments here
    parser.add_argument(
        "--data_path",
        type=str,
        default="/hpcwork/thes1906/Foundation_Models/Top_Quark_Tagging_Ref_Data/QCD_train_discrete_pT_eta_phi.h5",
        help="Path to training data file",
    )
    parser.add_argument(
        "--num_const", type=int, default=50, help="Number of constituents"
    )
    parser.add_argument(
        "--num_jets_train", type=int, default=False, help="Number of jets to train on"  # False = Complete Trainingset
    )
    parser.add_argument(
        "--num_jets_val", type=int, default=False, help="Number of jets to validate on"
    )
    parser.add_argument(
        "--add_start",
        action="store_true",
        help="Whether to use a start particle (learn first particle as well)",
    )
    parser.set_defaults(add_start = True)

    parser.add_argument(
        "--add_stop",
        action="store_true",
        help="Whether to use a end particle (learn jet length as well)",
    )
    parser.set_defaults(add_stop = True)
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dim of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--n_pt", type= int, default = 40, help = "Number of pt bins")
    parser.add_argument("--n_eta", type= int, default = 30, help = "Number of eta bins")
    parser.add_argument("--n_phi", type= int, default = 30, help = "Number of phi bins")
    parser.add_argument("--causal_mask", action = "store_true", help = "Wether to use a causal mask in the attention layer.")
    parser.set_defaults(causal_mask = True)
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for storing logs and model files",
        default = "/hpcwork/thes1906/Ph.D./transformer_output"
    )
    parser.add_argument("--name", type=str, default = "latest", help = "Name of model")
    parser.add_argument("--contin", "-c", action = "store_true", help = "if selected training is continued with specified file, all args are ignored and taken from original run")
    parser.set_defaults(contin = False )
    parser.add_argument("--batch_size", type=int, default = 5)

    args = parser.parse_args()
    return args

#saves a model to disk
def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))

def load_model(model_path):
    model = torch.load(model_path)

    return model

#just use for testing
def load_model_checkpoint(checkpoint_path):

    num_features = 3

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    train_args = checkpoint["args"]
    print(f"Args used for training: {train_args}")

    newModel = model.JetTransformer(
        hidden_dim=train_args["hidden_dim"],
        num_layers=train_args["num_layers"],
        num_heads=train_args["num_heads"],
        num_features=num_features,
        num_bins=(train_args["n_pt"], train_args["n_eta"], train_args["n_phi"]),
        dropout=train_args["dropout"],
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        causal_mask = train_args["causal_mask"],
    )

    newModel.load_state_dict(checkpoint["model_state"])

    return newModel

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, path="/hpcwork/thes1906/Ph.D./transformer_output/checkpoints", name = "latest"):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "args":vars(args)
    }

    torch.save(checkpoint, os.path.join(path, name + ".pt"))
