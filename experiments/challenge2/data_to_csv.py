from pathlib import Path
import numpy as np
from tqdm import tqdm
from eegchallenge import config


def main():
    scratch = config.arrays_dir(2)
    X=np.load(scratch/"X.npy",mmap_mode="r"); y=np.load(scratch/"y.npy",mmap_mode="r")
    assert len(X.shape)==3
    X = X.reshape(-1,X.shape[1]*X.shape[2])
    print(X.shape)
    print(y.shape)
    n=int(0.9*X.shape[0])
    def dump(path, Xs, ys, bs=5000):
        with open(path,"w") as f:
            for i in tqdm(range(0, Xs.shape[0], bs)):
                np.savetxt(f, np.c_[ys[i:i+bs], Xs[i:i+bs]], delimiter=",")
    dump(scratch/"train.csv", X[:n], y[:n]); dump(scratch/"valid.csv", X[n:], y[n:])


if __name__=='__main__':
    main()
