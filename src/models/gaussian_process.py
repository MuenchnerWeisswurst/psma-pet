import torch
import torch.nn as nn
import gpytorch


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Sequential
                 ):
        super(FeatureExtractor, self).__init__()

        self.net = nn.Sequential(*model[:-1])
        self.num_features = model[-1].in_features

    def forward(self, img: torch.Tensor):
        return self.net(img)


class StochasticVariationalGaussianProcess(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        variational_dist = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_dist,
                                                                        learn_inducing_locations=True)

        super(StochasticVariationalGaussianProcess, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVDKLGP(gpytorch.Module):
    def __init__(self, inducing_points: torch.Tensor,
                 feature_extractor: FeatureExtractor
                 ):
        super(SVDKLGP, self).__init__()
        self.gplayer = StochasticVariationalGaussianProcess(inducing_points)
        self.features = feature_extractor

    def forward(self, x):
        x_f = self.features(x)
        return self.gplayer(x_f)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    device = "cpu"
    f_0 = torch.ones(1, 10).to(device)
    f_1 = -torch.ones(1, 10).to(device)
    f = torch.stack([f_0, f_1])
    print(f.shape)
    targets = torch.Tensor([0, 1]).to(device)
    densenet = torch.nn.Sequential(
        torch.nn.Conv1d(1, 2, (7,), (3,)),
        torch.nn.AdaptiveAvgPool1d((1,)),
        torch.nn.Flatten()
    )
    ind_p = torch.rand(4, 2) * 2 - 1
    lgp = SVDKLGP(ind_p, densenet).to(device)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    mll = gpytorch.mlls.VariationalELBO(likelihood, lgp.gplayer, num_data=2)
    lr = 0.1
    ngd = gpytorch.optim.NGD([{'params': lgp.gplayer.variational_parameters()}], lr=0.1, num_data=2)
    optim = torch.optim.Adam([
        {'params': lgp.features.parameters()},
        {'params': lgp.gplayer.hyperparameters(), 'lr': lr * 0.01},
        {'params': likelihood.parameters()},
    ], lr=lr, weight_decay=0)
    miter = tqdm(range(4000))
    losses = []
    with gpytorch.settings.num_likelihood_samples(8):
        for i in miter:
            ngd.zero_grad()
            optim.zero_grad()
            o = lgp(f)
            loss = -mll(o, targets)
            losses.append(loss.item())
            miter.set_postfix(loss=loss.item())
            loss.backward()
            ngd.step()
            optim.step()

        preds = lgp(f)
        print(likelihood(preds).mean)
        print(likelihood(preds).sample())

    print(lgp.features(f))
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()
