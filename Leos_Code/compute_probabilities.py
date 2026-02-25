import torch
from helpers_compute_probabilities import *
import model
import time
from tqdm import tqdm
import numpy as np
import dataset
from torch.utils.data import DataLoader
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def probabilities(
        model,
        dataloader,
        device,
        args,
        train_args,
):

    n_jets = args.n_jets

    start_time = time.time()

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    with torch.no_grad():
            progress_bar = tqdm(dataloader, desc= "Computing probabilities", leave = False)

            output_file = open(args.output_file, mode="w", newline="")
            writer = csv.writer(output_file)

            writer.writerow(["probs","multiplicity"])

            for x in progress_bar:
                x = x.to(device)

                valid_mask = ~((x == -1).any(dim=-1))   
                seq_lens = valid_mask.sum(dim = 1)

                #create logits from forward pass
                logits = model.forward(x)

                #compute actual probabilities
                probs = model.probability(logits, x, logarithmic = True)

                result = torch.stack((probs, seq_lens), dim = 1)

                writer.writerows(result.tolist())

    total_time = time.time() - start_time
    print(f"\nFinished calculating probabily of {n_jets} jets")
    print(f"Total time: {total_time:.2f} s")
    print(f"Average speed: {n_jets / total_time:.2f} jets/s")    

if __name__ == "__main__":
    args = parse_inputs()

    print(f"Running on device: {device}")

    num_features = 3

    ##load model from file
    print(f"Load model state from:{args.model_path}")

    checkpoint = torch.load(args.model_path)

    train_args = checkpoint["args"]
    print(f"Args used for training: {train_args}")

    sampleModel = model.JetTransformer(
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
    sampleModel.to(device)

    sampleModel.load_state_dict(checkpoint["model_state"])

    sampleModel.eval()

    #load datasets
    test_loader = DataLoader(dataset.JetDataSet(
        data_dir = args.data_path,
        tag = "test",
        num_features=num_features,
        num_bins=(train_args["n_pt"], train_args["n_eta"], train_args["n_phi"]),
        num_const=args.num_const,
        add_start=train_args["add_start"],
        add_stop=train_args["add_stop"],
        n_jets=args.n_jets,
        key = args.input_key,
        ),
        batch_size=args.batch_size)

    print(f"Test set size: {len(test_loader)}")

    ## run sampling
    probabilities(sampleModel, test_loader, device, train_args=train_args, args = args)
    