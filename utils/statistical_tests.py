import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns


def calculate_correlation(A, B):
    """Calculates correlation between two vectors A and B

    :param A: first vector
    :type A: list, numpy.ndarray
    :param B: second vector
    :type B: list, numpy.ndarray
    :return: correlation between A and B
    :rtype: float
    """
    return np.corrcoef(A, B)[0, 1]

def calculate_covariance(A, B):
    """Calculates covariance between two vectors A and B

    :param A: first vector
    :type A: list, numpy.ndarray
    :param B: second vector
    :type B: list, numpy.ndarray
    :return: covariance between A and B
    :rtype: float
    """
    return np.cov(A, B)[0, 1]

def calculate_spearman_correlation(A, B):
    """Calculates Spearman correlation

    :param A: first vector
    :type A: list, numpy.ndarray
    :param B: second vector
    :type B: list, numpy.ndarray
    :return: Spearman correlation coefficient and p-value
    :rtype: tuple
    """
    coef, p_value = spearmanr(A, B)
    return coef, p_value


def scatterplot(A, B, A_name, B_name, path):
    """Creates a scatterplot of two vectors A and B

    :param A: first vector
    :type A: list, numpy.ndarray
    :param B: second vector
    :type B: list, numpy.ndarray
    :param A_name: name of the first vector
    :type A_name: str
    :param B_name: name of the second vector
    :type B_name: str
    :param path: path to save the scatterplot
    :type path: str
    """
    plt.scatter(A, B)
    plt.xlabel(A_name)
    plt.ylabel(B_name)
    plt.savefig(path)
    plt.close()

def regplot(A, B, A_name, B_name, path):
    """Creates a regression plot of two vectors A and B

    :param A: first vector
    :type A: list, numpy.ndarray
    :param B: second vector
    :type B: list, numpy.ndarray
    :param A_name: name of the first vector
    :type A_name: str
    :param B_name: name of the second vector
    :type B_name: str
    :param path: path to save the regression plot
    :type path: str
    """
    sns.regplot(x=A, y=B)
    plt.xlabel(A_name)
    plt.ylabel(B_name)
    plt.savefig(path)
    plt.close()


