
import cv2
import numpy as np
import os
import sys

import matplotlib.pyplot as plt


reinit = False
ITERATION = 5

# P = [0,4,8,12,16,17,18,19]
P = [i for i in range(0,25,2)]
x_ticks = []
test_acc_avg = []
test_acc_random_avg = []
for _ite in P:
    x_ticks.append(f"{100*0.8**_ite:.1f}")
    test_acc_scatter = []
    test_acc_random_scatter = []
    x = []
    for ITE in range(ITERATION):
        log_file = f"log/{ITE}/train_{_ite}_normal.log"
        log_file2 = f"log/{ITE}_reinit/train_{_ite}_reinit.log"
        with open(log_file, "r") as f:
            data = f.readlines()
        train_iter = []
        test_acc = []
        for line in data:
            _line = line.split("\t")
            train_iter.append(int(_line[1][1:-1]))
            test_acc.append(float(_line[-1].split(" ")[-2]))
        test_acc_scatter.append(test_acc[-1])
        # test_acc_scatter.append(max(test_acc))
        with open(log_file2, "r") as f:
            data = f.readlines()
        test_acc = []
        for line in data:
            _line = line.split("\t")
            train_iter.append(int(_line[1][1:-1]))
            test_acc.append(float(_line[-1].split(" ")[-2]))
        test_acc_random_scatter.append(test_acc[-1])
        x.append(_ite)
    test_acc_avg.append(sum(test_acc_scatter)/len(test_acc_scatter))
    test_acc_random_avg.append(sum(test_acc_random_scatter)/len(test_acc_random_scatter))
    # plt.scatter(x, test_acc_scatter, c="blue", marker="|")
plt.plot(P, test_acc_avg, c="blue", label="winning tickets")
plt.plot(P, test_acc_random_avg, c="red", label="random sampling")
plt.title("Test Accuracy vs Pruning Rate (mnist) (AVG of 5 trials)")
plt.xlabel("pruning rate")
plt.ylabel("test accuracy")
plt.xticks(P, x_ticks)
# plt.xticks([100*0.8**i for i in [0,4,8,12,16,20]])
plt.legend()
plt.ylim(90, 100)
plt.grid(color="gray")
plt.savefig("fig/fig1.png")
plt.close()

P = [i for i in range(0,25,3)]
for _ite in P:
    test_acc_list = []
    for ITE in range(ITERATION):
        log_file = f"log/{ITE}/train_{_ite}.log"
        if reinit:
            log_file = f"log/{ITE}/train_{_ite}_reinit.log"
        with open(log_file, "r") as f:
            data = f.readlines()[3:]
        train_iter = []
        test_acc = []
        for line in data:
            _line = line.split("\t")
            train_iter.append(int(_line[1][1:-1]))
            test_acc.append(float(_line[-1].split(" ")[-2]))
        test_acc_list.append(test_acc)

    x = np.array([i for i in range(len(test_acc))])
    test_acc = np.array(test_acc_list).mean(axis=0)
    e = np.array(test_acc_list).var(axis=0)
    plt.plot(x, test_acc, label=f"{100*0.8**_ite:.1f}")
    # plt.errorbar(x, test_acc, yerr=e, lw=0.5)

plt.title("training step vs test accuracy  (mnist)")
plt.xlabel("iteration")
plt.ylabel("test accuarcy")
plt.grid(color="gray")
plt.legend()
plt.savefig(f"fig/fig2.png")

plt.ylim(95, 99)
plt.savefig(f"fig/fig2_0.png")

plt.close()



for _ite in range(25):
    test_acc_list = []
    for ITE in range(ITERATION):
        log_file = f"log/{ITE}/train_{_ite}.log"
        if reinit:
            log_file = f"log/{ITE}/train_{_ite}_reinit.log"
        with open(log_file, "r") as f:
            data = f.readlines()[3:]
        train_iter = []
        test_acc = []
        for line in data:
            _line = line.split("\t")
            train_iter.append(int(_line[1][1:-1]))
            test_acc.append(float(_line[-1].split(" ")[-2]))
        test_acc_list.append(test_acc)

    x = np.array([i for i in range(len(test_acc))])
    test_acc = np.array(test_acc_list).mean(axis=0)
    e = np.array(test_acc_list).var(axis=0) # ; print(test_acc.shape, e.shape)
    plt.plot(x, test_acc, label="average")
    # plt.errorbar(x, test_acc, yerr=e, lw=0.5)
    plt.title(f"{100*0.8**_ite:.1f} training step vs test accuracy  (mnist)")
    plt.xlabel("iteration")
    plt.ylabel("test accuarcy")
    plt.grid(color="gray")
    plt.legend()
    plt.savefig(f"fig/iteration/{_ite}.png")
    plt.close()
