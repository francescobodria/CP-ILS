from __future__ import print_function

from .PDASH_utils import HeuristicSetSelection

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DIExplainer(ABC):
    """
    DIExplainer is the base class for Directly Interpretable unsupervised explainers (DIE).
    Such explainers generally rely on unsupervised techniques to explain datasets and model predictions.
    Examples include DIP-VAE[#1]_, Protodash[#2]_, etc.

    References:
        .. [#1] Variational Inference of Disentangled Latent Concepts from Unlabeled Observations (DIP-VAE), ICLR 2018.
         Kumar, Sattigeri, Balakrishnan. https://arxiv.org/abs/1711.00848
        .. [#2] ProtoDash: Fast Interpretable Prototype Selection, 2017.
        Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi.
        https://arxiv.org/abs/1707.01212
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize a DIExplainer object.
        ToDo: check common steps that need to be distilled here.
        """

    @abc.abstractmethod
    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, *argv, **kwargs):
        """
        Explain the data or model.
        """
        raise NotImplementedError



class ProtodashExplainer(DIExplainer):
    """
    ProtodashExplainer provides exemplar-based explanations for summarizing datasets as well
    as explaining predictions made by an AI model. It employs a fast gradient based algorithm
    to find prototypes along with their (non-negative) importance weights. The algorithm minimizes the maximum
    mean discrepancy metric and has constant factor approximation guarantees for this weakly submodular function. [#]_.

    References:
        .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
           "ProtoDash: Fast Interpretable Prototype Selection"
           <https://arxiv.org/abs/1707.01212>`_
    """

    def __init__(self):
        """
        Constructor method, initializes the explainer
        """
        super(ProtodashExplainer, self).__init__()

    def set_params(self, *argv, **kwargs):
        """
        Set parameters for the explainer.
        """
        pass

    def explain(self, X, Y, m, kernelType='other', sigma=2, optimizer='cvxpy'):
        """
        Return prototypes for data X, Y.

        Args:
            X (double 2d array): Dataset you want to explain.
            Y (double 2d array): Dataset to select prototypical explanations from.
            m (int): Number of prototypes
            kernelType (str): Type of kernel (viz. 'Gaussian', / 'other')
            sigma (double): width of kernel
            optimizer (string): qpsolver ('cvxpy' or 'osqp')
            
        Returns:
            m selected prototypes from X and their (unnormalized) importance weights
        """
        return( HeuristicSetSelection(X, Y, m, kernelType, sigma, optimizer) )
