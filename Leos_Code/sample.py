import torch
from helpers_sample import *
import model
import h5py
import time
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(
        model,
        device,
        args,
        train_args,
):

    n_jets = args.n_jets

    progress_bar = tqdm(total = n_jets, 
                        desc = f"Sampling Jets",
                        )
    
    flat_dim = 3 * args.max_length

    # Create axis0 labels once
    axis0_labels = []
    for i in range(args.max_length):
        axis0_labels.extend([f"PT_{i}", f"ETA_{i}", f"PHI_{i}"])

    with h5py.File(args.output_file, "w") as f:

        #prepare dataset
            
        dset = f.create_dataset(
            "sampled_jets",
            shape=(n_jets, 3 * args.max_length),
            maxshape=(None, 3 * args.max_length),
            dtype=np.int16,
            chunks=(min(args.batch_size, n_jets), 3 * args.max_length)
        )
        # Speichere Feature-Namen als Attribut
        dset.attrs["feature_names"] = np.bytes_(axis0_labels)

        jets_written = 0
        start_time = time.time()

        while jets_written < n_jets:

            current_batch = min(args.batch_size, n_jets - jets_written)

            start_time_batch = time.time()

            jets = model.sample(
                batch_size = current_batch,
                max_length = args.max_length,
                temperature = args.temperature,
                topk = args.topk,
                device = device
            ).cpu().numpy()

            #flattens them into right shape for h5 file
            jets_flat = jets.reshape(current_batch, -1)
            #jet_ids = np.arange(jets_written, jets_written + current_batch)

            dset[jets_written:jets_written + current_batch, :] = jets_flat

            jets_written += current_batch

            progress_bar.update(current_batch)

            dt = time.time() - start_time_batch
            speed = current_batch / dt
            progress_bar.set_postfix({"jets/s": f"{speed:.2f}"})

    progress_bar.close()

    total_time = time.time() - start_time
    print(f"\nFinished sampling {n_jets} jets")
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

    try:
        sampleModel = torch.compile(sampleModel)
        print("\nModel successfully compiled.")
    except Exception as e:
        print("\ntorch.compile failed, running without compile:", e)

    sampleModel.load_state_dict(checkpoint["model_state"])

    sampleModel.eval()

    ## run sampling
    with torch.no_grad():
        sample(sampleModel, device, args, train_args,)
    