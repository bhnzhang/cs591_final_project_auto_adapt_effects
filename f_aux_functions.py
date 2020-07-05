"""
helper functions
"""

# -------------------------
# Imports
# -------------------------

# numpy
import numpy as np

import math

import scipy.io.wavfile as wf

from scipy import signal

from collections.abc import Iterable

# -------------------------
# constants
# -------------------------

SR = 44100 # sampling rate

# -------------------------
# function defs
# -------------------------

def readWaveFile(infile):
    rate, data = wf.read(infile)
    return data
    #
    # print(data.shape)
    #
    # return np.array([data[:,0], data[:,1]])

def writeWaveFile(filename, data, rate=SR ):
    dataAlt = (data.T).astype(np.int16)
    wf.write(filename, rate, data.T)


def make_sin_LFO(n_samps, rate, amplitude):
    """
    Generate a sinusoid

    Inputs:
        n_samps
            number of samples
        rate
            oscillation frequency, in units of 1/samples
        amplitude
            amplitude
    :return:
    """
    return amplitude * np.sin(2 * np.pi * rate * np.arange(n_samps))

def make_triangle_LFO(n_samps, rate, amplitude):
    """
    Generate a triangle wave

    Inputs:
        n_samps
            number of samples
        rate
            oscillation frequency, in units of 1/samples
        amplitude
            amplitude
    :return:
    """
    return amplitude * signal.sawtooth(2 * np.pi * rate * np.arange(n_samps), width = 0.5)

def is_iterable(x):
    """
    returns whether x is an iterable type or not
    :param x:
    :return:
    """
    return isinstance(x, Iterable)