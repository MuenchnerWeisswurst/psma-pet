import abc
from abc import ABC

import torch


class BaseRegressionNormalizer(ABC):
    """
    Abstract class for regression target normalizer
    """
    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalization function
        :param x: input tensor
        :return:
        """
        return torch.empty()

    @abc.abstractmethod
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalization function
        :param x: normalized input tensor
        :return:
        """
        return torch.empty()


class LogScaleNormalizer(BaseRegressionNormalizer):
    def __init__(self, targets: torch.Tensor):
        self.targets = targets
        self.log_targets = torch.log(self.targets)
        self.mean = self.log_targets.mean()
        self.std = self.log_targets.std()

    def normalize(self, targets: torch.Tensor):
        t_log = torch.log(targets)
        return (t_log - self.mean) / self.std

    def inverse(self, normed_targets: torch.Tensor):
        t_un = (normed_targets + self.mean) * self.std
        return torch.exp(t_un)


class LogScaleNormalizerNoInverse(BaseRegressionNormalizer):
    def __init__(self, targets: torch.Tensor):
        self.normalizer = LogScaleNormalizer(targets)

    def normalize(self, targets: torch.Tensor):
        return self.normalizer.normalize(targets)

    def inverse(self, normed_targets: torch.Tensor):
        return normed_targets


class MeanStdNormalizer(BaseRegressionNormalizer):
    def __init__(self, targets: torch.Tensor):
        self.targets = targets
        self.mean = self.targets.mean()
        self.std = self.std.std()

    def normalize(self, targets: torch.Tensor):
        return (targets - self.mean) / self.std

    def inverse(self, normed_targets: torch.Tensor):
        t_un = (normed_targets + self.mean) * self.std
        return t_un


class LinearNormalizer(BaseRegressionNormalizer):
    def __init__(self):
        pass

    def normalize(self, x: torch.Tensor):
        return x

    def inverse(self, x: torch.Tensor):
        return x


class ThresholdNormalizer(BaseRegressionNormalizer):
    def __init__(self):
        pass

    def normalize(self, x: torch.Tensor):
        x[x > 1] = 1
        return x

    def inverse(self, x: torch.Tensor):
        x[x > 1] = 1
        return x


class LogNormalizer(BaseRegressionNormalizer):
    def __init__(self):
        pass

    def normalize(self, x: torch.Tensor):
        return torch.log(x)

    def inverse(self, x: torch.Tensor):
        return torch.exp(x)


class LogNormalizerNoInverse(BaseRegressionNormalizer):
    def __init__(self):
        pass

    def normalize(self, x: torch.Tensor):
        return torch.log(x)

    def inverse(self, x: torch.Tensor):
        return x
