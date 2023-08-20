from .beta import BetaLikelihood
from .bernoulli import BernoulliLikelihood
from .continous_bernoulli import ContinousBernoulliLikelihood
from .categorical import CategoricalLikelihood
from .normal import NormalLikelihood
from .normal_mean import *
from .heterogeneous import HeterogeneousLikelihood
from .delta import DeltaLikelihood

likelihood_dict = {
    "beta": BetaLikelihood,
    "ber": BernoulliLikelihood,
    "cb": ContinousBernoulliLikelihood,
    "cat": CategoricalLikelihood,
    "normal": NormalLikelihood,
    "normal1": NormalMean1Likelihood,
    "normal01": NormalMean01Likelihood,
    "het": HeterogeneousLikelihood,
    "delta": DeltaLikelihood,
}


def build_likelihoods_list(lik_info_list):
    likelihoods = []
    for (lik_name_i, domain_size_i) in lik_info_list:
        lik = likelihood_dict[lik_name_i](domain_size_i)
        likelihoods.append(lik)
    return likelihoods
