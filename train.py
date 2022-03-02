import argparse

import numpy as np
import torch
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from matplotlib import pyplot as plt

from data import _get_dataparams_complex, _get_dataparams_simple
from pfLDA import pfLDA
from utils import accuracy, visualize_topics, check_and_load_data_LDA


def main(args):
    _check_and_load_data = lambda split: check_and_load_data_LDA(split, args.dataroot, args.dataname, args.batch_size)

    loader_train, loader_val, loader_test = list(map(_check_and_load_data, ['train', 'val', 'test']))

    def train_pfLDA(p, lr_exponent):
        I, V, K = loader_train.dataset.I, loader_train.dataset.V, args.K
        optim_hyperparams = {"epochs_train": args.num_epochs, "lr_train": 10**lr_exponent, "epochs_val": 1000, "lr_val": 5e-2}
        model = pfLDA({"M": args.M, "I": I, "V": V, "K": args.K, "p": p})
        # Get the true params
        fn = _get_dataparams_simple if args.dataname == "simple" else _get_dataparams_complex
        K, beta, eta, theta = fn(I, V)

        model.set_params(beta=beta, eta=eta, varphi=torch.ones_like(model.logitvarphi.data))
        model.beta.requires_grad_(False)
        model.eta.requires_grad_(False)
        model.logitvarphi.requires_grad_(False)
        # model.gamma.requires_grad_(False)
        ELBOs_train, ELBOs_val = model.fit(loader_train, loader_val, optim_hyperparams)
        import pdb;
        pdb.set_trace()

        fig, ax = plt.subplots(2, 1)
        plot_fname = f"topics_pfLDA_p_{p:.3f}_lr_{10**lr_exponent:.3f}_K_{K}_{args.dataname}.pdf"
        visualize_topics(model.beta.softmax(-1).data.numpy(), fname=None, ax=ax[0])
        visualize_topics(model.pi.softmax(-1).data.numpy().reshape(1,-1), fname=None, ax=ax[1])
        fig.suptitle(f"K = {K}")
        fig.savefig(plot_fname)
        print(f"Saved topics to {plot_fname}")

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, 11) * (args.num_epochs // 10), np.array(ELBOs_train[1:]), marker='x', label="Random Init")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train ELBO")
        ax.legend()
        ax.set_title("Train ELBO Learning Curve and Ground Truth")
        fig.savefig(f"ELBO_curve_pfLDA_p_{p:.3f}_lr_{10**lr_exponent:.3f}_K_{K}_{args.dataname}.pdf")
        return accuracy(model.predict_prob(loader_val.dataset.W, optim_hyperparams), loader_val.dataset.Y)

    if args.bayesopt:
        load_prev_results = False
        logspath = f"./logs_{args.dataname}.json"

        pbounds = {'p': (1, 1), 'lr_exponent': (-4, 0)}
        hyperparam_optimizer = BayesianOptimization(
            f=train_pfLDA,
            pbounds=pbounds,
            random_state=1,
            verbose=2
        )

        if load_prev_results: load_logs(hyperparam_optimizer, logs=[logspath])

        logger = JSONLogger(path=logspath, reset=(not load_prev_results))

        hyperparam_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        hyperparam_optimizer.maximize(
            init_points=(0 if load_prev_results else 3),
            n_iter=30,
        )
    else:
        train_pfLDA(args.p, np.log10(args.lr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--K", type=int, default=2, help="Number of topics")
    parser.add_argument("--M", type=int, default=5, help="Number of MCMC samples for reparameterization")
    parser.add_argument("--p", type=float, default=0.25, help="Value for the switch prior for pf-sLDA")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-1, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--bayesopt", action="store_true", help="Flag to use bayesopt for hyperparameter optimization") #False by default
    parser.add_argument("--dataroot", type=str,
                        default="/Users/abhisheksharma/PycharmProjects/pfLDA/data/simulated",
                        help="Root path of data splits to which dataname_{train, val, test}.pkl are appended")
    parser.add_argument("--dataname", type=str, default="simple",
                        help="Data name where DATANAME_{train, val, test}.pkl are the data files")

    args = parser.parse_args()

    main(args)