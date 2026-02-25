import torch
import os
from model_new import JetTransformer
from helpers_train import *
from torch.utils.data import DataLoader
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    torch.backends.cuda.enable_flash_sdp(True)   # erlaubt Flash SDP (beta-Flag)
    print("flash_sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
except Exception as e:
    print("Could not enable flash_sdp:", e)

def train(model, train_loader, val_loader, optimizer, scheduler, args,
          epochs = 10,
          ):
    model.train()
    model.to(device)

    best_val_loss = float("inf")

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        total_train_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{epochs} [Training]",
            leave = True
        )
        for x in progress_bar:
            #move batch to gpu if possible
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = model.loss(logits, x)
            
            #backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            #update progress bar
            progress_bar.set_postfix(loss = loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        ### run validation after epoc
        avg_val_loss = validate(model, val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} finished | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, args, name = args.name + "_best", path=os.path.join(args.output_path, "checkpoints"))
            print(f"Checkpoint saved as: {args.name}_best.pt")

        ### logging
        log_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        df = pd.DataFrame([log_data])

        df.to_csv(
            os.path.join(args.output_path, f"{args.name}_training_log.csv"),
            mode="a",
            header=not os.path.exists(os.path.join(args.output_path, f"{args.name}_training_log.csv")),
            index=False
        )
        #df.to_hdf(
        #    os.path.join(args.output_path, f"{args.name}_training_log.h5"),
        #    mode="w", key="df"
        #)

        #writer.add_scalar("Loss/train", avg_train_loss, epoch)
        #writer.add_scalar("Loss/val", avg_val_loss, epoch)
        #writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

#for running the validation set
def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc= "Validation", leave = False)

        for x in progress_bar:
            x = x.to(device)

            logits = model(x)
            loss = model.loss(logits, x)

            total_loss += loss.item()
            progress_bar.set_postfix(val_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    args = parse_inputs()

    # writer = SummaryWriter(args.output_path)

    print("Running trainings process:")
    print(f"Running on device: {device}")

    num_features = 3
    #load datasets
    print(f"Loading training set")
    train_loader = DataLoader(JetDataSet(
        data_dir = args.data_path,
        tag = "train",
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        num_jets=args.num_jets_train,
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=args.batch_size,
        num_workers=2,           # num_cpus_per_task-2
        pin_memory=True,
        persistent_workers=True)

    print(f"Loading validation set")
    val_loader = DataLoader(JetDataSet(
        data_dir = args.data_path.replace("train", "val"),
        tag = "val",
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        num_jets=args.num_jets_val,
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=args.batch_size,
        num_workers=2,      
        pin_memory=True,
        persistent_workers=True)

    #construct model
    model = JetTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        dropout=args.dropout,
        add_start=args.add_start,
        add_stop=args.add_stop,
        causal_mask = args.causal_mask,
    )

    model.to(device)

    try:
        model = torch.compile(model)
        print("\nModel successfully compiled.")
    except Exception as e:
        print("\ntorch.compile failed, running without compile:", e)

    #add optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr,
    )

    #create scheduler
    scheduler = lr_scheduler(
        optimizer,
        warmup_steps=int(0.1*len(train_loader)*args.num_epochs),
        total_steps=len(train_loader)*args.num_epochs
    )

    #print(train_loader.dataset[:, : , :])

    train(model, train_loader, val_loader, optimizer, scheduler, args, epochs=args.num_epochs)


