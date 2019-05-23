
import numpy as np
import torch

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def original_initialization(model, state_dict):
    for name, p in model.named_parameters():
        if "weight" in name or "weight_prune" in name:
            # print("a", name)
            m = name.split(".")[0]
            p.data = model.state_dict()[f"{m}.mask"] * state_dict[name]
        if "bias" in name:
            # print("b", name)
            p.data = model.state_dict()[name]

