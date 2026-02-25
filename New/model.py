
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from argparse import ArgumentParser
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import warnings
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class FactorizedOutputHead(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 num_features = 3,
                 num_bins = (42, 32, 32) #+start and stop no padding
                 ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.num_bins = num_bins

        self.sum_bins = int(np.sum(num_bins))
        #add single linear layer for embedding(d = 256) --> (d = 256 * sum_bins)
        self.linear = nn.Linear(hidden_dim, hidden_dim * self.sum_bins)

        self.activation = nn.Softplus() #constraints the output of the singlelayer to always positive / similiar to Relu but smooth
        self.logit_rescale = nn.Parameter(torch.tensor(1.0)) #adds a single learnable parameter for the rescaling of the final logit

    def forward(self, h):
        batch_size, num_const, _hidden_dim = h.shape 
        assert _hidden_dim == self.hidden_dim

        #project hidden state (d = 256) into (d = 256 * sum_bins) with linear layer
        bin_emb_flat = self.activation(self.linear(h))

        #reshape state (d = 256 * sum_bins) into matrix with ( sum_bins x 256)
        bin_emb = bin_emb_flat.view(batch_size, num_const, self.sum_bins, self.hidden_dim)
        #split it into 3 different matrices for pt, eta and phi along dimension with size of sum_bins
        pt_emb, eta_emb, phi_emb = torch.split(bin_emb, self.num_bins, dim = 2)

        #now recombine into a single particle vector
        #first get a single vector for all pt, eta combinations so 42*32 possibilities
        pt_eta_emb = pt_emb.unsqueeze(2) * eta_emb.unsqueeze(3) # shape = (42 x 32 x 256)
        #reshape into ((42*32)x256)
        pt_eta_emb = pt_eta_emb.view(batch_size, num_const, -1, self.hidden_dim) #shape= ((42*32)x256)

        #combine with phi into final logit
        logits_3d = pt_eta_emb @ phi_emb.transpose(2, 3) #((42*32)x256)@(256 x 32) = ((42*32)x32)
        #flatten again and scale --> shape = batchsize x num_const x (42*32*32)

        logits = self.logit_rescale.exp() * logits_3d.view( batch_size, num_const, -1 )
        #logits have dimension n_pt*n_eta*n_phi
        return logits

class JetTransformer(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 num_layers = 8,
                 num_heads = 4,
                 num_features = 3,
                 num_bins = (40, 30, 30), #--> (41, 31, 31)
                 dropout = 0.1,
                 add_start = True,
                 add_stop = True,
                 causal_mask = True):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_features = num_features
        self.num_bins = num_bins
        self.causal_mask = causal_mask

        if add_start:
            self.num_bins = [x + 1 for x in self.num_bins]
            print("Added start token!")
        if add_stop:
            self.num_bins = [x + 1 for x in self.num_bins]
            print("Added stop token!")

        print(f"Num_bins: {self.num_bins}")

        num_bins = self.num_bins

        #this total voc size includes all physical bins + start + stop + pad --> (43,33,33)
        #this is only used for the input embedding --> the output does not make room for padding because we dont want to predict padding
        self.total_voc_bins = [x + 2 for x in self.num_bins] #+2 because num_bin describes the max value (incl) so 0...40 (41) + start + stop + pad = 44 bins
        self.total_voc_size = np.prod(self.total_voc_bins)

        self.voc_bins = [x + 1 for x in self.num_bins]
        self.voc_size = np.prod(self.voc_bins)

        print(f"Total_voc_bin: {self.total_voc_bins}")

        #chooses pad bin to total_bins +1 in each feature 
        self.PAD_BIN = [bins + 1 for bins in self.num_bins]
        self.START_BIN = [0 for i in range(self.num_features)]
        self.STOP_BIN = self.num_bins

        self.PAD_IDX = -1

        print(f"Bins reserved for PAD: {self.PAD_BIN}")

        #----- adding all different layers to the network
        #we want to embedd all features seperately so we a embedding layer with 3 parrallel embeddings
        self.feature_embeddings = nn.ModuleList(
            [
                nn.Embedding(embedding_dim=self.hidden_dim, num_embeddings=self.total_voc_bins[i])
                for i in range(num_features)
            ]
        )

        #add the #num_layers of TransformerEncoder Layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model = hidden_dim, #expected input dimension
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        #this is only used during training and randomly drops a percentage (0.1) of the neurons
        #this prevents overfitting, so the model learns not to rely on specific single neurons
        self.dropout = nn.Dropout(dropout)
        #--- > (256 DIM EMBEDDING)
        self.output_layer = FactorizedOutputHead(hidden_dim = self.hidden_dim,
                                                 num_features = self.num_features,
                                                 num_bins = self.voc_bins, #here we do not count the padding bin
                                                )

        #---> logit (42*32*32) DIM 
        #define output layer
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.PAD_IDX)

    def forward(self,
                x, #has shape (batch_size, num_const, num_features) ---> logits with shape (batch_size, num_const, total_bins)
                ):

        batch_size = x.shape[0]
        num_const = x.shape[1] #saves number of const. used in this data
        num_features = x.shape[2] 
        assert num_features == self.num_features

        #creates a padding_mask to mask invalid particles
        padding_mask = (x[:, :, 0] < 0) #shows every where an invalid particle is does not exclude start and stop needs shape (batch_size, num_const)

        #replaces the -1 as padding bin with a valid pad bin which is stop_bin + 1
        for i in range(self.num_features):
            x[:, : , i] = torch.where(x[:, :, i] < 0, self.PAD_BIN[i], x[:, :, i])

        #extract features from x
        #embed seperately and add together 
        # (padding tokens are mapped to a padding embedding but never used because padding mask will exclude them)
        # they are also excluded while calculating the loss
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        #create a casual mask
        attention_mask = None
        if self.causal_mask:
            #creates a triangular casual mask
            attention_mask = torch.triu(
                torch.ones(num_const, num_const, device = x.device),
                diagonal = 1
            ).bool()

        #transformer layer
        for transformer_layer in self.layers:
            emb = transformer_layer(src = emb, 
                                    src_mask = attention_mask, 
                                    src_key_padding_mask=padding_mask,
                                    is_causal = self.causal_mask
                                    )
        
        #apply last norm
        emb = self.out_norm(emb)
        #apply last droptout
        emb = self.dropout(emb)

        #print("forward: Transformer layer + out_norm + dropout successfull!")
        #project to final logit
        logits = self.output_layer(emb)
        #print("forward: Output layer successfull!")

        return logits

    #this computes autoregressive loss for gpt style training
    def loss(self, logits_joint, targets):
        #computes cross entropy over full dictonary of size prod(num_bins)
        #ignores padding -1

        #discard the final logit, there is no target for this logit
        logits_joint = logits_joint[:,:-1].reshape(-1, self.voc_size) 

        #compute target ids from target tokens
        target_ids = self.tuple_to_index(targets[..., 0], targets[..., 1], targets[..., 2], self.num_bins)

        #shift targets to the right, because t_id1 contains what logit_0 should predict
        target_ids = target_ids[:, 1:].reshape(-1) 
        #print("------ target_ids-----")
        #print(target_ids)

        #this computes cross entropy loss only if the target particle is valid (ignores the padding because our network does not produce padding)
        #this now computes the autoregressive loss we need for the gpt style loss
        loss = self.criterion(logits_joint, target_ids)

        return loss
    
    #returns the probability of a target jet relative to the logits of the network
    #ideally the logits come from a forward pass from the same targets
    # (thats basically the same from what the loss function is calculated)
    def probability(
            self,
            logits_joint, 
            targets, 
            logarithmic = False,
            topk = None,

    ):
        batch_size, seq_length, voc_size = logits_joint.shape

        #discard the final logit, there is no target for this logit
        logits_joint = logits_joint[:,:-1]
        #apply softmax to get probs from logits
        probs = torch.softmax(logits_joint, dim = -1)

        # --- TOP-K FILTERING ---
        if topk is not None:
            # find topk probs at each (B,S)
            topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

            # mask: True for entries in topk
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)

            # suppress everything else
            probs = probs.masked_fill(~mask, 1.0)

        #compute target ids from target tokens
        target_ids = self.tuple_to_index(targets[..., 0], targets[..., 1], targets[..., 2], self.num_bins)
        
        #shift targets to the right, because t_id1 contains what logit_0 shWould predict
        target_ids = target_ids[:, 1:]

        padding_mask = target_ids == self.PAD_IDX
        #compute the prob that each particle id in target_ids shows up
        target_ids = target_ids.masked_fill(padding_mask, 0)
        probs = probs.gather(
            dim = -1, #gather along last dimension
            index= target_ids.unsqueeze(-1)
        ).squeeze(-1)

        probs = probs.masked_fill(padding_mask, 1.0)

        if logarithmic:
            prob = torch.log(probs).sum(dim = -1)
        else:
            prob = probs.prod(dim = -1)

        return prob

    def tuple_to_index(self, pt, eta, phi, num_bins):
        """
        Map (pt, eta, phi) bin indices into single vocabulary index.

        and padding -1 to -1 in index

        index = pt + n_pt * eta + (n_pt*n_eta) * phi
        """
        padding_mask = (pt == -1)

        joint_id = pt + eta * num_bins[0] + (num_bins[0]*num_bins[1])* phi
        joint_id[padding_mask] = self.PAD_IDX

        return joint_id

    def index_to_tuple(self, index, num_bins):
        """
        Inverse mapping: dictionary index -> (pt, eta, phi)
        """

        padding_mask = (index == self.PAD_IDX)

        pt = index % num_bins[0]
        eta = (index // num_bins[0]) % num_bins[1]
        phi = index // (num_bins[0] * num_bins[1])

        pt[padding_mask] = -1
        eta[padding_mask] = -1
        phi[padding_mask] = -1

        return pt, eta, phi
    
def parse_inputs():
    parser = ArgumentParser()

    #add arguments here
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to training data file",
    )
    parser.add_argument(
        "--num_const", type=int, default=50, help="Number of constituents"
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
        default = "output/"
    )
    parser.add_argument("--name", type=str, default = "latest", help = "Name of model")
    parser.add_argument("--contin", "-c", action = "store_true", help = "if selected training is continued with specified file, all args are ignored and taken from original run")
    parser.set_defaults(contin = False )
    parser.add_argument("--batch_size", type=int, default = 100)

    args = parser.parse_args()
    return args

#saves a model to disk
def save_model(model, log_dir, name):
    torch.save(model, os.path.join(log_dir, f"model_{name}.pt"))

def load_model(model_path):
    model = torch.load(model_path)

    return model

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, path="output/checkpoints", name = "latest"):
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

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):

    # Scheduler: ensure T_max >= 1
    cosine_T = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    return scheduler

#creates a torch dataset from a preprocessed h5 file
class JetDataSet(Dataset):
    def __init__(self, data_dir, tag : str, 
                 num_features=3,
                 num_bins = (40, 30, 30),
                 num_const = 50,
                 add_stop = True,
                 add_start = True,
                 ):
        df = pd.read_hdf(data_dir, key = "df")
        self.data = disc_to_token(df,
                                  num_features=num_features,
                                  num_bins=num_bins,
                                  num_const=num_const,
                                  add_end=add_stop,
                                  add_start=add_start)
        self.tag = tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#takes in binned discretized data as a dataframe and outputs tokens as torch tensor that can be used for training/validating or other purposes
def disc_to_token(df, 
                  num_features, #number of different features(pt, eta, phi)
                  num_bins,
                  num_const, #number of constituents per jet, this can be limited
                  to_tensor = True, #if we want to return a torch tensor
                  add_start = True, #wether to add start and end tokens
                  add_end = True, 
                  ): 
    
    x = df.to_numpy(dtype = np.int64)[:, : num_const * num_features] # this keeps only as many constituents we want in our data
    x = x.reshape(x.shape[0], -1, num_features) #this reshapes the data such that its 3 dimensional with [njets, nconst, nfeatures] [[[pt_1, eta_1, phi_1], ...],[[pt_1, eta_1, phi_1], ...],...]
    
    x = x.copy()

    padding_mask = x == -1 #marks every where a invalid const is

    #add start and stop token if needed
    if add_start: 
        #this shifts every valid bin --> 1 so the 0 can now be the start token
        x[~padding_mask] += 1
        
        #this adds a start particle with (0,0,0) to the start of every jet
        x = np.concatenate(
            (
                np.zeros((len(x), 1, num_features), dtype=int),
                x,
            ),
            axis=1,
        )

        num_bins= [x +1 for x in num_bins]
        print("Added start token. New bins are now:", num_bins)
    #add stop token only if the actual number of const. in the jet is smaller than the limit we have set for const.
    #so if a jet fills all the const dont set a stop token
    if add_end:
        num_bins= [x +1 for x in num_bins]
        
        #compute length of each jet
        jet_length = (~padding_mask[:, :, 0]).sum(1) + 1 #this gives the index of the first invalid const. +1 because of start token
        valid = (jet_length >= 0) & (jet_length < x.shape[1]) #this ensures that the index we want to set to the stop token is not out of bounds 

        x[np.arange(x.shape[0])[valid], jet_length[valid]] = num_bins        

        print("Added stop token. New bins are now:", num_bins)

    if to_tensor: 
        x = torch.tensor(x)

    return x

def train(model, train_loader, val_loader, optimizer, scheduler, args,
          epochs = 10,
          ):
    model.train()
    model.to(device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_train_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc = f"Epoch {epoch+1}/{epochs} [Training]",
            leave = True
        )
        for x in progress_bar:
            #move batch to gpu if possible
            x = x.to(device)

            #compute one forward pass and the loss on the data passed into the network
            logits = model(x)
            loss = model.loss(logits, x) #the target is the data we have trained with
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

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
    
    # Remove specific warning about epoch parameter in scheduler.step() 
    warnings.filterwarnings("ignore", message=".*epoch parameter in `scheduler.step\\(\\)` was not necessary.*")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_inputs()

    writer = SummaryWriter(args.output_path)

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
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=args.batch_size)

    print(f"Loading validation set")
    val_loader = DataLoader(JetDataSet(
        data_dir = args.data_path.replace("train", "val"),
        tag = "val",
        num_features=num_features,
        num_bins=(args.n_pt, args.n_eta, args.n_phi),
        num_const=args.num_const,
        add_stop=args.add_stop,
        add_start=args.add_start
        ),
        batch_size=args.batch_size)

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

    #add optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr = args.lr,
    )

    #create scheduler
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_steps=int(0.1*len(train_loader)*args.num_epochs),
        total_steps=len(train_loader)*args.num_epochs
    )

    #print(train_loader.dataset[:, : , :])

    train(model, train_loader, val_loader, optimizer, scheduler, args, epochs=args.num_epochs)















