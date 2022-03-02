import pickle
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import einsum
from torch.distributions import Dirichlet, Categorical, Multinomial, Bernoulli
from torch.nn.functional import normalize

from utils import visualize_topics, visualize_coefficients

f_type = torch.float32


def _get_dataparams_simple(I, V=9):
    K = 3
    # There are 3 "True Topics" (First 2 are predictive of Y)
    # Theta for first topic is -vely correlated with Y
    # Theta for next topic is +vely correlated with Y
    # Last topic is uncorrelated with Y
    # First 6 words are relevant
    beta = torch.zeros(K, V, dtype=f_type)
    beta[0, :V // 3] = 1
    beta[1, V // 3:2 * (V // 3)] = 1
    beta[2, 2 * (V // 3):] = 1
    beta = normalize(beta, p=1, dim=-1)
    print(f"true beta: \n{beta.data.numpy().round(2)}")
    eta = torch.tensor([-10, 10., 0], dtype=f_type)
    torch.random.manual_seed(42)
    theta = Dirichlet(concentration=1. * torch.ones(K, dtype=f_type)).sample((I,))
    return K, beta, eta, theta


def _get_dataparams_complex(I, V=9):
    K = 7
    assert V > 6
    noisy = False
    # There are 7 "True Topics" (First 4 are predictive of Y)
    # Theta for first 2 is -vely correlated with Y
    # Theta for next 2 is +vely correlated with Y
    # Last 3 are uncorrelated with Y
    # First 4 words are relevant
    beta = torch.zeros(K, V, dtype=f_type)
    if noisy: beta += 0.1
    beta[0, 0] = 1
    beta[1, 1] = 1
    beta[2, 2] = 1
    beta[3, 3] = 1
    beta[4, 4] = 1
    beta[5, 5] = 1
    beta[6, 6:] = 1
    beta = normalize(beta, p=1, dim=-1)
    print(f"true beta: \n{beta.data.numpy().round(2)}")
    eta = torch.tensor([-5, -5, 5, 5, 0, 0, 0], dtype=f_type)
    torch.random.manual_seed(42)
    theta = Dirichlet(concentration=1. * torch.ones(K, dtype=f_type)).sample((I,))
    return K, beta, eta, theta


def simulate_data_simple(I, J, V):
    K, beta, eta, theta = _get_dataparams_simple(I, V)

    Z = Categorical(probs=theta).sample((J,)).transpose(0, 1)
    _W = Multinomial(1, probs=beta).sample((I, J))
    assert Z.shape == (I, J)
    assert _W.shape == (I, J, K, V)

    W = torch.empty(I, V, dtype=f_type)

    for i in range(I):
        for v in range(V):
            W[i] = (_W[i][torch.arange(J), Z[i]]).sum(0)

    Y = Bernoulli(probs=einsum("k,ik", eta, theta).sigmoid()).sample()

    assert W.shape == (I, V)
    assert Y.shape == (I,)

    # visualize_topics(beta.numpy(), "topics-simple.pdf")
    fig, ax = plt.subplots(2, 1)
    visualize_coefficients(eta.data.numpy().reshape(1, -1), ax=ax[0])
    visualize_topics(beta.numpy(), fname=None, ax=ax[1])
    ax[0].set_ylabel("Topics")
    ax[1].set_xlabel("Topics")
    fig.suptitle(f"{K} Topics, {V} Words")

    fname = "topics-simple.pdf"
    fig.savefig(fname)
    print(f"Saved topics to {fname}")

    # print(W, Y)
    return W, Y


def simulate_data_complex(I, J, V):
    K, beta, eta, theta = _get_dataparams_complex(I, V)
    # theta = Dirichlet(concentration=1. * torch.ones(K, dtype=f64)).sample((I,))

    Z = Categorical(probs=theta).sample((J,)).transpose(0, 1)
    _W = Multinomial(1, probs=beta).sample((I, J))
    assert Z.shape == (I, J)
    assert _W.shape == (I, J, K, V)

    W = torch.empty(I, V, dtype=f_type)

    for i in range(I):
        for v in range(V):
            W[i] = (_W[i][torch.arange(J), Z[i]]).sum(0)

    Y = Bernoulli(probs=einsum("k,ik", eta, theta).sigmoid()).sample()

    assert W.shape == (I, V)
    assert Y.shape == (I,)

    fig, ax = plt.subplots(2, 1)
    visualize_coefficients(eta.data.numpy().reshape(1, -1), ax=ax[0])
    visualize_topics(beta.numpy(), fname=None, ax=ax[1])

    ax[0].set_ylabel("Topics")
    ax[1].set_xlabel("Topics")
    fig.suptitle(f"{K} Topics, {V} Words")
    fname = "topics-complex.pdf"
    fig.savefig(fname)
    print(f"Saved topics to {fname}")

    return W, Y


def _split_train_val_test(W, Y):
    assert len(W) == len(Y)
    W_splits = np.array_split(W, 3)
    Y_splits = np.array_split(Y, 3)
    return [{'W': _W_split, 'Y': _Y_split} for _W_split, _Y_split in zip(W_splits, Y_splits)]


def save_data(dataname):
    I, J, V = 6, 1000, 9
    fn = simulate_data_simple if dataname == "simple" else simulate_data_complex

    W, Y = fn(I, J, V)

    # dataset_train = LDADataset(W[I // 3:], Y[I // 3:])
    # dataset_val = LDADataset(W[:I // 3], Y[:I // 3])

    data_splits = _split_train_val_test(W, Y)
    for split, data_split in zip(["train", "val", "test"], data_splits):
        pkl_fname = Path(f"./data/simulated/{dataname}_{split}.pkl")
        with open(pkl_fname, "wb") as f:
            pickle.dump(data_split, f)


if __name__ == '__main__':
    save_data("simple")
    # save_data("complex")
