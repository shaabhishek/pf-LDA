import argparse
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import normalize

from data import _get_dataparams_complex, _get_dataparams_simple
from pfLDA import pfLDA
from utils import check_and_load_data_LDA, visualize_topics, visualize_coefficients

f_type = torch.float32


def compare_ELBO_incorrect_K(args):
    _check_and_load_data = lambda split: check_and_load_data_LDA(split, args.dataroot, args.dataname, args.batch_size)

    loader_train, loader_val, loader_test = list(map(_check_and_load_data, ['train', 'val', 'test']))

    I_train, K_model, V = loader_train.dataset.I, args.K, loader_train.dataset.V

    # Get the true params
    if args.dataname == "simple":
        assert K_model == 2
        K_data, beta_data, eta_data, _ = _get_dataparams_simple(I_train, V)
        beta_true = torch.zeros(K_model, V, dtype=f_type)
        beta_true[:2] = beta_data[:2]
        eta_true = torch.zeros(K_model, dtype=f_type)
        eta_true[0] = eta_data[0]
        eta_true[1] = eta_data[1]
        varphi_true = torch.zeros(V, dtype=f_type)
        varphi_true[:6] = 1
    else:
        assert K_model == 2
        K_data, beta_data, eta_data, _ = _get_dataparams_complex(I_train, V)
        beta_true = torch.zeros(K_model, V, dtype=f_type)
        beta_true[0] = normalize(beta_data[:2].sum(0), p=1, dim=-1)
        beta_true[1] = normalize(beta_data[2:4].sum(0), p=1, dim=-1)
        eta_true = torch.zeros(K_model, dtype=f_type)
        eta_true[0] = eta_data[0]
        eta_true[1] = eta_data[2]
        varphi_true = torch.zeros(V, dtype=f_type)
        varphi_true[:4] = 1

    def train_pfLDA(p, lr_exponent):
        optim_hyperparams = {"epochs_train": args.num_epochs, "lr_train": 10**lr_exponent, "epochs_val": 1000, "lr_val": 5e-2}
        model = pfLDA({"M": args.M, "I": I_train, "V": V, "K": args.K, "p": p})
        ELBOs_train, ELBOs_val = model.fit(loader_train, loader_val, optim_hyperparams)

        # Create new model with the right params
        newmodel = pfLDA({"M": args.M, "I": I_train, "V": V, "K": args.K, "p": p})
        newmodel.set_params(beta=beta_true, eta=eta_true, varphi=varphi_true)
        list(map(lambda p: p[1].requires_grad_(False), filter(lambda p: p[0] in ['beta', 'eta', 'logitvarphi'], newmodel.named_parameters())))

        # Re-train the variational params
        newoptim_hyperparams = deepcopy(optim_hyperparams)
        newoptim_hyperparams.update({"lr_train": .2})
        ELBOs_train_true, ELBOs_val_true = newmodel.fit(loader_train, loader_val, newoptim_hyperparams)
        ELBO_train_true = newmodel.ELBO_from_dataloader(loader_train, mean=True).data.numpy().item()
        print(f"True Train ELBO: {ELBO_train_true}")
        print(newmodel)

        fig, ax = plt.subplots(3, 1)
        plot_fname = f"topics_pfLDA_p_{p:.3f}_lr_{10**lr_exponent:.3f}_K_{K_model}_{args.dataname}.pdf"
        visualize_coefficients(model.eta.data.numpy().reshape(1, -1), ax=ax[0])
        visualize_topics(model.beta.softmax(-1).data.numpy(), fname=None, ax=ax[1])
        visualize_topics(model.pi.softmax(-1).data.numpy().reshape(1, -1), fname=None, ax=ax[2])
        fig.suptitle(f"K = {K_model}")
        fig.savefig(plot_fname)
        print(f"Saved topics to {plot_fname}")

        fig, ax = plt.subplots(3, 1)
        plot_fname = f"topics_trueinit_pfLDA_p_{p:.3f}_lr_{10**lr_exponent:.3f}_K_{K_model}_{args.dataname}.pdf"
        visualize_coefficients(newmodel.eta.data.numpy().reshape(1, -1), ax=ax[0])
        visualize_topics(newmodel.beta.softmax(-1).data.numpy(), fname=None, ax=ax[0])
        visualize_topics(newmodel.pi.softmax(-1).data.numpy().reshape(1, -1), fname=None, ax=ax[1])
        fig.suptitle(f"K = {K_model}")
        fig.savefig(plot_fname)
        print(f"Saved topics to {plot_fname}")

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, 11) * (args.num_epochs // 10), np.array(ELBOs_train[1:]), marker='x', label="Random Init")
        ax.plot(np.arange(1, 11) * (args.num_epochs // 10), np.array(ELBOs_train_true[1:]), marker='o', label="True Init")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train ELBO")
        ax.legend()
        ax.set_title("Train ELBO Learning Curve and Ground Truth")
        fig.savefig(f"ELBO_curve_{args.dataname}.pdf")

    train_pfLDA(args.p, np.log10(args.lr))

def compare_ELBO_correct_K(args):
    _check_and_load_data = lambda split: check_and_load_data_LDA(split, args.dataroot, args.dataname, args.batch_size)

    loader_train, loader_val, loader_test = list(map(_check_and_load_data, ['train', 'val', 'test']))

    def train_pfLDA(p, lr_exponent):
        I_train, K, V = loader_train.dataset.I, args.K, loader_train.dataset.V
        optim_hyperparams = {"epochs_train": args.num_epochs, "lr_train": 10**lr_exponent, "epochs_val": 1000, "lr_val": 5e-2}
        model = pfLDA({"M": args.M, "I": I_train, "V": V, "K": args.K, "p": p})
        ELBOs_train, ELBOs_val = model.fit(loader_train, loader_val, optim_hyperparams)

        # Get the true params
        fn = _get_dataparams_simple if args.dataname == "simple" else _get_dataparams_complex
        K, beta, eta, theta = fn(I_train, V)

        # Create new model with the right params
        newmodel = pfLDA({"M": args.M, "I": I_train, "V": V, "K": args.K, "p": p})
        newmodel.set_params(beta=beta, eta=eta, varphi=torch.ones_like(newmodel.logitvarphi.data))
        list(map(lambda p: p[1].requires_grad_(False), filter(lambda p: p[0] in ['beta', 'eta', 'logitvarphi'], newmodel.named_parameters())))

        # Re-train the variational params
        newoptim_hyperparams = deepcopy(optim_hyperparams)
        # newoptim_hyperparams.update({"epochs_train": 10, "lr_train": .2})
        newoptim_hyperparams.update({"lr_train": .2})
        ELBOs_train_true, ELBOs_val_true= newmodel.fit(loader_train, loader_val, newoptim_hyperparams)
        ELBO_train_true = newmodel.ELBO_from_dataloader(loader_train, mean=True).data.numpy().item()
        print(f"True Train ELBO: {ELBO_train_true}")
        print(newmodel)

        fig, ax = plt.subplots()
        ax.plot(np.arange(1,11)*(args.num_epochs//10), np.array(ELBOs_train[1:]), marker='x', label="Random Init")
        ax.plot(np.arange(1,11)*(args.num_epochs//10), np.array(ELBOs_train_true[1:]), marker='o', label="True Init")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train ELBO")
        ax.legend()
        ax.set_title("Train ELBO Learning Curve and Ground Truth")
        fig.savefig(f"ELBO_curve_{args.dataname}.pdf")

        return None

    train_pfLDA(p=args.p, lr_exponent=np.log10(args.lr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--K", type=int, default=2, help="Number of topics")
    parser.add_argument("--M", type=int, default=5, help="Number of MCMC samples for reparameterization")
    parser.add_argument("--p", type=float, default=0.25, help="Value for the switch prior for pf-sLDA")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-1, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size")
    parser.add_argument("--bayesopt", action="store_true",
                        help="Flag to use bayesopt for hyperparameter optimization")  # False by default
    parser.add_argument("--dataroot", type=str,
                        default="/Users/abhisheksharma/PycharmProjects/pfLDA/data/simulated",
                        help="Root path of data splits to which dataname_{train, val, test}.pkl are appended")
    parser.add_argument("--dataname", type=str, default="simple",
                        help="Data name where DATANAME_{train, val, test}.pkl are the data files")

    args = parser.parse_args()

    compare_ELBO_correct_K(args)
    # compare_ELBO_incorrect_K(args)