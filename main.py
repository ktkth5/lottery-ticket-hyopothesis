
import argparse

import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from  models import lanet
import utils

#endless loader
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#logger
from datetime import datetime
def logging(fname, text):
    f = open(fname,"a")
    out_text = "[%s]\t%s" %(str(datetime.now()),text)
    f.write(out_text+"\n")
    # print (out_text)
    f.close()


def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resample = True if args.prune_type=="resample" else False
    reinit = True if args.prune_type=="reinit" else False

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    model = lanet.LeNet(mask=True).to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())
    torch.save({"state_dict": initial_state_dict}, f"saves/initial_state_dict_{args.prune_type}.pth.tar")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(f"saves/model_{args.prune_type}.pth.tar")
        start_iteration = checkpoint["iter"]
        model.load_state_dict(checkpoint["state_dict"])
        initial_state_dict = torch.load(f"saves/initial_state_dict_{args.prune_type}.pth.tar")["state_dict"]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ITERATION = 25
    for _ite in range(start_iteration, ITERATION):
        # Pruning
        if not _ite == 0:
            model.prune_by_percentile(resample=resample, reinit=reinit)
            if not reinit:
                utils.original_initialization(model, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"--- Pruning Level [{ITE}:{_ite}/25]: {1.0*(0.8**(_ite))*100:.1f}---")
        utils.print_nonzeros(model)

        pbar = tqdm(range(args.end_iter))
        for iter_ in pbar:
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader)
                logging(f"log/{ITE}_reinit/train_{_ite}_{args.prune_type}.log", "[%06d]\tAccuracy = %04f " % (iter_, float(accuracy)))
            loss = train(model, train_loader, optimizer)
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}%')

        torch.save({"state_dict": model.state_dict(), "iter": _ite}, f"saves/model_{args.prune_type}.pth.tar")


def train(model, train_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer.zero_grad()

    model.train()
    imgs, targets = next(train_loader)
    imgs, targets = imgs.to(device), targets.to(device)
    output = model(imgs)
    train_loss = F.nll_loss(output, targets)
    train_loss.backward()

    # zero-out all the gradients corresponding to the pruned connections
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        grad_tensor = p.grad.data.cpu().numpy()
        grad_tensor = np.where(tensor == 0, 0, grad_tensor)
        p.grad.data = torch.from_numpy(grad_tensor).to(device)
    optimizer.step()
    return train_loss.item()


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20000, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--valid_freq", default=100, type=int)
    parser.add_argument("--prune_type", default="normal", type=str, help="normal | resample | reinit")

    args = parser.parse_args()

    for i in range(0, 5):
        main(args, ITE=i)

    # tensor = np.random.randn(10)
    # tensor[0] = 0
    # grad_tensor = np.random.randn(10)
    # print("tesnsor\n", tensor)
    # print("grad_tensor\n", grad_tensor)
    # print("where\n",np.where(tensor==0,0,grad_tensor))

