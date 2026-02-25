from argparse import ArgumentParser
import os

def parse_inputs():

    parser = ArgumentParser()
    #### add arguments here
    parser.add_argument("--model_path", type = str, help = "Path to the model file")
    parser.add_argument("--n_jets", type = int, default = 100, help = "Number of sampled jets")
    parser.add_argument("--max_length", type = int, default = 100, help = "Max length of a generated jet" )
    parser.add_argument("--batch_size", type = int, default = 10, help = "Number of jets sampled together")
    parser.add_argument("--topk", type = int, help = "If set particles get only sampled from the <topk> most probable")

    args = parser.parse_args()
    return args