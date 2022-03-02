import torch
from torch import nn, einsum, logsumexp, optim
from torch.distributions import transform_to, Dirichlet, Categorical, Bernoulli, Normal
from torch.distributions.utils import probs_to_logits
from torch.nn.functional import softplus, log_softmax
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader

from data import simulate_data_simple
from utils import multi_get, LDADataset, collate_fn_tensor, accuracy

f_type = torch.float32


class pfLDA(nn.Module):
    """
    Hyperparameters:
        M: Number of MCMC samples to approximate expectation using reparameterization trick
        K: Number of topics
        V: Number of codes
        p: switch prior

    Data Parameters:
        I: Number of patients
        J[i]: Number of codes observed for patient i

    Parameters (all probabilities are normalized / transformed to unconstrained axis for SGD to work):
        eta: regression parameters for P(Y | \Theta)
        beta: relevant topic log probabilities. shape: K x V
        pi: irrelevant topic log probabilities. shape: K
        alpha: alpha is per-patient topic distribution prior

        gamma: variational posterior params for theta (Dirichlet Distribution). shape: I x K
        phi: variational posterior params for Z (Categorical Distribution). shape: I x J x K
        varphi: variational posterior params for S (Bernoulli Distribution). shape: V

    """

    def __init__(self, hyperparams):
        super(pfLDA, self).__init__()
        init_normal = lambda *shape: torch.randn(*shape, dtype=f_type)

        self.M, self.K, self.V, self.I, p = multi_get(hyperparams, ["M", "K", "V", "I", "p"])
        self.logitp = probs_to_logits(torch.tensor(p, dtype=f_type), is_binary=True)

        self.eta = nn.Parameter(init_normal(self.K))
        self.beta = nn.Parameter(init_normal(self.K, self.V))
        self.pi = nn.Parameter(init_normal(self.V))
        # self.alpha = nn.Parameter(init_normal(1))
        self.alpha = torch.tensor([1.], dtype=f_type).log()

        self.gamma = nn.Parameter(init_normal(self.I, self.K))
        self.phi = nn.Parameter(init_normal(self.I, self.V, self.K))
        self.logitvarphi = nn.Parameter(init_normal(self.V))

        # Regularizers
        self.prior_eta = Normal(torch.zeros(self.K), scale=torch.ones(self.K))
        self.prior_topics = Dirichlet(torch.ones(self.V))

    def get_normalized_params(self, I_vals: torch.Tensor) -> dict:
        """Return all parameters in (-\inf, \inf)"""

        dirichlet_concentration_transform = transform_to(Dirichlet.arg_constraints['concentration'])

        params = {
            "alpha": dirichlet_concentration_transform(self.alpha),
            "lbeta": log_softmax(self.beta, dim=-1),
            "lpi": log_softmax(self.pi, dim=-1),
            "gamma": dirichlet_concentration_transform(self.gamma[I_vals]),
            "lphi": log_softmax(self.phi[I_vals], dim=-1),
            "logitvarphi": self.logitvarphi,
            "eta": self.eta,
        }
        return params

    def ELBO_old(self, W_batch: torch.Tensor, Y_batch: torch.Tensor, patient_idxs_batch: torch.Tensor) -> torch.Tensor:
        BS = len(W_batch)

        params = self.get_normalized_params(patient_idxs_batch)
        alpha, gamma, eta, lphi, lbeta, lpi, logitvarphi = multi_get(params, ["alpha", "gamma", "eta", "lphi", "lbeta", "lpi", "logitvarphi"])
        K, M = multi_get(self.__dict__, ["K", "M"])

        digamma_tilde = torch.digamma(gamma) - torch.digamma(gamma.sum(1, keepdims=True))
        assert digamma_tilde.shape == (BS, K)

        term_1 = BS * (K * torch.lgamma(alpha) - torch.lgamma(K * alpha)) + (alpha - 1) * digamma_tilde.sum()

        theta_dist = Dirichlet(gamma)
        theta_samples = theta_dist.sample((M,))
        term_2 = einsum("i,ik,k->", Y_batch, theta_dist.mean, eta) - softplus(
            einsum("k,mik->mi", eta, theta_samples)).sum() / M

        term_3 = einsum("iv,ivk,ik->", W_batch, lphi.exp(), digamma_tilde)

        term_4 = self.logitp * einsum("iv,v->", W_batch, logitvarphi.sigmoid()) - W_batch.sum() * softplus(self.logitp)

        term_5 = einsum("iv,ivk,kv,v->", W_batch, lphi.exp(), lbeta, logitvarphi.sigmoid())
        # print(term_5)

        term_6 = einsum("v,v,iv->", (1 - logitvarphi.sigmoid()), lpi, W_batch)

        term_7 = theta_dist.entropy().sum()\
                 + einsum("iv,iv->", W_batch, Categorical(logits=lphi).entropy())\
                 + einsum("iv,v->", W_batch, Bernoulli(logits=logitvarphi).entropy())

        # reg_eta = ds.Dirichlet(self.alpha * torch.ones(self.K)).log_prob(Theta.softmax(-1))
        logprior_eta = self.prior_eta.log_prob(eta).sum()
        logprior_beta = self.prior_topics.log_prob(lbeta.exp()).sum()
        logprior_pi = self.prior_topics.log_prob(lpi.exp())

        ELBO_batch = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + logprior_eta + logprior_beta + logprior_pi
        return ELBO_batch

    def ELBO(self, W_batch: torch.Tensor, Y_batch: torch.Tensor, patient_idxs_batch: torch.Tensor) -> torch.Tensor:
        BS = len(W_batch)

        params = self.get_normalized_params(patient_idxs_batch)
        alpha, gamma, eta, lbeta, lpi, logitvarphi = multi_get(params, ["alpha", "gamma", "eta", "lbeta", "lpi", "logitvarphi"])
        K, M = multi_get(self.__dict__, ["K", "M"])

        digamma_tilde = torch.digamma(gamma) - torch.digamma(gamma.sum(1, keepdims=True))
        assert digamma_tilde.shape == (BS, K)

        term_1 = BS * (K * torch.lgamma(alpha) - torch.lgamma(K * alpha)) + (alpha - 1) * digamma_tilde.sum()

        theta_dist = Dirichlet(gamma)
        theta_samples = theta_dist.sample((M,))

        term_2 = einsum("i,ik,k->", Y_batch, theta_dist.mean, eta) - softplus(einsum("k,mik->mi", eta, theta_samples)).sum() / M

        term_3 = self.logitp * einsum("iv,v->", W_batch, logitvarphi.sigmoid()) - W_batch.sum() * softplus(self.logitp)

        term_4 = einsum("v,v,iv->", (1 - logitvarphi.sigmoid()), lpi, W_batch)

        # term_5 = einsum("iv,iv,v->", W_batch, logsumexp(theta_samples.unsqueeze(-1).log() + lbeta, dim=-2).mean(0), logitvarphi.sigmoid())
        term_5 = einsum("iv,iv,v->", W_batch, einsum("mik,kv->miv", theta_samples, lbeta.exp()).log().mean(0), logitvarphi.sigmoid())
        # term_5_upperbd = einsum("iv,iv,v->", W_batch, einsum("ik,kv->iv", gamma, lbeta.exp()).log(), logitvarphi.sigmoid())
        print(term_5)

        term_6 = theta_dist.entropy().sum()\
                 + einsum("iv,v->", W_batch, Bernoulli(logits=logitvarphi).entropy())

        logprior_eta = self.prior_eta.log_prob(eta).sum()
        logprior_beta = self.prior_topics.log_prob(lbeta.exp()).sum()
        logprior_pi = self.prior_topics.log_prob(lpi.exp())

        ELBO_batch = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + logprior_eta + logprior_beta + logprior_pi
        # print(term_1, term_2, term_3, term_4, term_5, term_6)
        return ELBO_batch

    def predict_prob(self, W: torch.Tensor, optim_hyperparams):
        assert len(W.shape) == 2
        alpha, eta, lbeta, lpi, logitvarphi = multi_get(self.get_normalized_params(torch.arange(0)), ["alpha", "eta", "lbeta", "lpi", "logitvarphi"])

        beta, pi, varphi = lbeta.exp(), lpi.exp(), logitvarphi.sigmoid()

        ltheta = self.compute_theta_MAP(W, beta, pi, varphi, optim_hyperparams)

        preds = einsum("ik,k->i", ltheta.softmax(-1), eta).sigmoid()
        return preds

    def compute_theta_MAP(self, W, beta, pi, varphi, optim_hyperparams: dict) -> torch.Tensor:
        BS = len(W)
        num_epochs, lr = multi_get(optim_hyperparams, ["epochs_val", "lr_val"])
        ltheta = torch.randn(BS, self.K, dtype=f_type, requires_grad=True)
        optimizer = Adam(params=[ltheta], lr=lr)
        for epoch_num in range(num_epochs + 1):
            optimizer.zero_grad()
            theta = ltheta.softmax(-1)
            obj = torch.mean((self.alpha - 1) * theta.sum() + einsum("iv,iv->i", W, torch.log(einsum("ik,kv->iv", theta, beta) * varphi + pi * (1 - varphi))))
            obj.mul(-1).backward(retain_graph=True)
            optimizer.step()
            if epoch_num % (num_epochs//2) == 0:
                print(f"Epoch {epoch_num}")
                print(f"Theta MAP Obj: {obj.data.numpy().item()}")
        return ltheta.data

    def fit(self, loader_train:DataLoader, loader_val:DataLoader, optim_hyperparams: dict):
        num_epochs, lr = multi_get(optim_hyperparams, ["epochs_train", "lr_train"])
        optimizer = optim.RMSprop(params=self.parameters(), lr=lr)
        # optimizer = LBFGS(params=self.parameters(), lr=lr)

        ELBOs_train = []
        ELBOs_val = []

        # def closure():
        #     elbo = torch.zeros(1, dtype=f_type)
        #     for W_batch, Y_batch, patient_idxs_batch in loader_train:
        #         optimizer.zero_grad()
        #         elbo += self.ELBO(W_batch, Y_batch, patient_idxs_batch)
        #     return elbo.mul(-1).div(len(loader_train.dataset))
        #
        # for epoch_num in range(num_epochs + 1):
        #     optimizer.zero_grad()
        #     optimizer.step(closure)


        for epoch_num in range(num_epochs + 1):
            for W_batch, Y_batch, patient_idxs_batch in loader_train:
                optimizer.zero_grad()
                elbo = self.ELBO(W_batch, Y_batch, patient_idxs_batch)
                elbo.mul(-1).div(len(W_batch)).backward()
                optimizer.step()


            if epoch_num % (num_epochs//10) == 0:
                with torch.no_grad():
                    ELBOs_train.append(self.ELBO_from_dataloader(loader_train, mean=True).data.numpy().item())
                    ELBOs_val.append(self.ELBO_from_dataloader(loader_val, mean=True).data.numpy().item())
                print(f"Epoch {epoch_num}")
                print(f"Mean Train ELBO: {ELBOs_train[-1]}")
                print(f"Mean Val ELBO: {ELBOs_val[-1]}")
                # print(f"Train Y Accuracy: {accuracy(einsum('k,ik->i', self.eta, self.gamma[patient_idxs_batch].exp()).sigmoid(), Y_batch)}")
                print(f"Train Y Accuracy: {accuracy(einsum('k,ik->i', self.eta, self.gamma.exp()).sigmoid(), loader_train.dataset.Y)}")
                print(f"Train Y Accuracy of Mean: {accuracy(loader_train.dataset.Y.mean(), loader_train.dataset.Y)}") #TODO: Fix this
                print(f"Val Y Accuracy: {accuracy(self.predict_prob(loader_val.dataset.W, optim_hyperparams), loader_val.dataset.Y)}")
                print(f"Val Y Accuracy of Mean: {accuracy(loader_val.dataset.Y.mean(), loader_val.dataset.Y)}") #TODO: Fix this
                print(self)

        return ELBOs_train, ELBOs_val

    def ELBO_from_dataloader(self, loader, mean=False) -> torch.Tensor:
        N = 0
        ELBO = torch.tensor([0.], dtype=f_type)
        for W_batch, Y_batch, patient_idxs_batch in loader:
            ELBO += self.ELBO(W_batch, Y_batch, patient_idxs_batch)
            N += len(patient_idxs_batch)

        if mean: ELBO /= N
        return ELBO




    def __repr__(self):
        out = ""
        alpha, eta, lbeta, lpi, logitvarphi = multi_get(self.get_normalized_params(torch.arange(0)), ["alpha", "eta", "lbeta", "lpi", "logitvarphi"])
        out += f"p: \n{self.logitp.sigmoid().item()} \n"
        out += f"alpha: \n{alpha.data.item()} \n"
        out += f"eta: \n{eta.data.numpy().round(2)} \n"
        out += f"beta: \n{lbeta.exp().data.numpy().round(2)} \n"
        out += f"pi: \n{lpi.exp().data.numpy().round(2)} \n"
        out += f"varphi: \n{logitvarphi.sigmoid().data.numpy().round(1)} \n"
        return out

    def set_params(self, beta=None, pi=None, alpha=None, eta=None, p=None, prior_eta=None, prior_topics=None, phi=None, gamma=None, varphi=None):
        with torch.no_grad():
            if beta is not None:
                # Assume input is in probability space
                self.beta.copy_(probs_to_logits(beta, is_binary=False))

            if pi is not None:
                # Assume input is in probability space
                self.pi.copy_(probs_to_logits(pi, is_binary=False))

            if alpha is not None:
                # Assume input is in Gamma parameter space (>0)
                self.alpha.copy_(alpha.log())

            if eta is not None:
                self.eta.copy_(eta)

            if p is not None:
                self.logitp.copy_(probs_to_logits(torch.tensor(p, dtype=f_type), is_binary=True))

            if prior_eta is not None:
                raise NotImplementedError

            if prior_topics is not None:
                raise NotImplementedError

            if phi is not None:
                # Assume input is in probability space
                self.phi.copy_(probs_to_logits(phi, is_binary=False))

            if gamma is not None:
                # Assume input is in Gamma parameter space (>0)
                self.gamma.copy_(gamma.log())

            if varphi is not None:
                # Assume input is in probability space
                self.logitvarphi.copy_(probs_to_logits(varphi, is_binary=True))


def test_pfLDA():
    V = 9
    I = 1000
    J = 1000
    M = 10

    K_model = 2
    p = 0.25

    W, Y = simulate_data_simple(I, J, V)
    # W, Y = simulate_data_complex(I, J, V)

    dataset_train = LDADataset(W[I//3:], Y[I//3:])
    dataset_val = LDADataset(W[:I//3], Y[:I//3])
    loader_train = DataLoader(dataset_train, batch_size=len(dataset_train)//5, shuffle=True, collate_fn=collate_fn_tensor)
    loader_val = DataLoader(dataset_val, batch_size=len(dataset_val)//5, shuffle=True, collate_fn=collate_fn_tensor)

    optim_hyperparams = {"epochs_train": 10000, "lr_train": 3e-2, "epochs_val": 1000, "lr_val": 3e-2}

    model = pfLDA({"M": M, "I": len(dataset_train), "V": V, "K": K_model, "p": p})
    print(model.fit(loader_train, loader_val, optim_hyperparams))


if __name__ == '__main__':
    test_pfLDA()

