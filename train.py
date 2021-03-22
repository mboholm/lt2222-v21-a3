import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from model import train
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def a(f):
    mm = []
    with open(f, "r") as q:
        for l in q:
            mm += [c for c in l]

    mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"]
    return mm, list(set(mm)) #M.B. (i) the text as a list of letters; (ii) the "vocabulary" (set of characers in the text)
 
def g(x, p):
    """
    M.B. Takes a list of symbols (x) and a "vocabulary" (p), and returns a collapsed one-hot representation of those symbols with regard to the vocabulary.
    """
    z = np.zeros(len(p))
    z[p.index(x)] = 1
    return z

def b(u, p):
    """

    Arguments: text (u); vocabulary (p)
    """

    gt = [] #M.B. this is the class
    gr = [] #M.B. this it the features
    for v in range(len(u) - 4):
        if u[v+2] not in vowels: #M.B. first two tokens are startsymbols ("<s>") 
            continue
        
        #M.B. "stops" at vowel v as defined by list "vowels" and takes ...
        h2 = vowels.index(u[v+2]) #M.B. the index of v (in list vowels) and 
        gt.append(h2) #M.B. appends to "gt" (list)
        r = np.concatenate([g(x, p) for x in [u[v], u[v+1], u[v+3], u[v+4]]]) #M.B. the "features" of v: two symbols to the left of v and two symbols to the right; 
        # feature representation as defined by g()
        gr.append(r) #M.B. and append to "gr" (list)

    return np.array(gr), np.array(gt) #M.B. train_x, train_y
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="k", type=int, default=200, help="Hidden layer size.")
    parser.add_argument("--r", dest="r", type=int, default=100, help="Epochs.")
    parser.add_argument("m", type=str, help="The file.")
    parser.add_argument("h", type=str, help="The path for model.")
    
    args = parser.parse_args()

    q = a(args.m)
    w = b(q[0], q[1])
    t = train(w[0], w[1], q[1], args.k, args.r)

    torch.save(t, args.h)
