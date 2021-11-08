import gpytorch
import torch

from src.data.target_functions import therapy_responsiveness, psa_pre_post_relation, psa_post, psa_pre

# Constant dict containing references to activation functions
ACTIVATIONS_MAP = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU,
    'prelu': torch.nn.PReLU,
}

# Constant dict containing references to target computing functions
TARGETS_MAP = {
    'therapy_responsiveness': therapy_responsiveness,
    'psa_pre_post_relation': psa_pre_post_relation,
    'psa_post': psa_post,
    'psa_pre': psa_pre,
}

# Constant dict containing references to loss functions
CRITERIONS_MAP = {
    'bce': torch.nn.BCEWithLogitsLoss,
    'mse': torch.nn.MSELoss,
    'elbo': gpytorch.mlls.VariationalELBO,
    'huber': torch.nn.HuberLoss,
}

# Constant dict containing references to optimizers
OPTIMIZER_MAP = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}

# Constant dict containing references to learning rate schedulers
SCHEDULER_MAP = {
    'reducelronplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
}

CMP_MAP = {
    'le': lambda x, y: x < y,
    'ge': lambda x, y: x > y,
}

LIKELIHOOD_MAP = {
    'gaussian': gpytorch.likelihoods.GaussianLikelihood,
    'bernoulli': gpytorch.likelihoods.BernoulliLikelihood
}
