"""
Digital filter implementations
"""

# -------------------------
# Imports
# -------------------------

import numpy as np

import math

# circular buffer class
from c_circ_buffer import Circ_buffer

# helper functions
from f_aux_functions import *

# distortion
from f_distortion import Distortion

# playing audio
# from IPython.display import Audio, display
# import sounddevice as sd
# import simpleaudio as sa

# plotting
import matplotlib.pyplot as plt


# -------------------------
# function defs
# -------------------------

def Biquad_Filter( X, a0, a1, a2, b1, b2, c0, d0 ):
    """
    Implements a generalized second order filter, with feed forward and feed back

    :param X:
    :param a0:
        feed forward, coeff of x(0)
    :param a1:
        feed forward coeff of x(-1)
    :param a2:
        feed forward coeff of x(-2)
    :param b1:
        feed backward coeff of -y(-1)
    :param b2:
        feed backward coeff of -y(-2)
    c0
        amount of wet
    d0
        amount of dry
    :return:
    """

    # make input and output buffers
    input_buff = Circ_buffer(buffer_size=4, delay_size=0)
    output_buff = Circ_buffer(buffer_size=4, delay_size=0)

    # make output vec
    output_sig = np.zeros(len(X)).astype(int)

    # process signal
    for ii in range(len(X)):

        # calculate output value
        output_val = X[ii]*a0 + \
                     ( (input_buff.read_value(2)*a2 - output_buff.read_value(2)*b2) + \
                     (input_buff.read_value(1)*a1 - output_buff.read_value(1)*b1) )

        output_sig[ii] = output_val
        input_buff.write_value(X[ii])
        output_buff.write_value(output_val)

    # end for loop

    output_sig = c0*output_sig + d0*X
    return output_sig

# end function Filter()

def LPF_firstorder( X, fc, rate=SR ):
    """
    First order low pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2 * np.pi * fc / rate
    gamma = np.cos(thetac) / (1 + np.sin(thetac))
    a0 = (1 - gamma) / 2
    a1 = (1 - gamma) / 2
    a2 = 0.0
    b1 = -gamma
    b2 = 0.0
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function LPF_firstorder()

def HPF_firstorder( X, fc, rate=SR ):
    """
    First order high pass filter
    not working for some reason

    :param X:
    :param fc:
        corner or cutoff freq
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2 * np.pi * fc / rate
    gamma = np.cos(thetac) / (1 + np.sin(thetac))
    a0 = (1.0 + gamma) / 2.0
    a1 = -(1.0 + gamma) / 2.0
    a2 = 0.0
    b1 = -gamma
    b2 = 0.0
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function HPF_firstorder()


def LPF_secondorder( X, fc, Q, rate=SR ):
    """
    Second order low pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    Q
        quality factor
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2 * np.pi * fc / rate
    d = 1/Q
    beta = 0.5 * (1 - (d/2)*np.sin(thetac))/(1 + (d/2)*np.sin(thetac))
    gamma = (0.5 + beta)*np.cos(thetac)
    a0 = (0.5 + beta - gamma)/2.0
    a1 = (0.5 + beta - gamma)
    a2 = (0.5 + beta - gamma)/2.0
    b1 = -2*gamma
    b2 = 2*beta
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function LPF_secondorder()


def HPF_secondorder( X, fc, Q, rate=SR ):
    """
    Second order high pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    Q
        quality factor
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2 * np.pi * fc / rate
    d = 1/Q
    beta = 0.5 * (1 - (d/2)*np.sin(thetac))/(1 + (d/2)*np.sin(thetac))
    gamma = (0.5 + beta)*np.cos(thetac)
    a0 = (0.5 + beta + gamma)/2.0
    a1 = -(0.5 + beta + gamma)
    a2 = (0.5 + beta + gamma)/2.0
    b1 = -2*gamma
    b2 = 2*beta
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function HPF_secondorder()

def BPF_secondorder( X, fc, Q, rate=SR ):
    """
    Second order band pass filter

    :param X:
    :param fc:
        corner or resonant freq
    Q
        quality factor
    :param rate:
        sampling rate
    :return:
    """

    K = np.tan( np.pi * fc / rate )
    delta = (K**2)*Q + K + Q
    a0 = K/delta
    a1 = 0.0
    a2 = -K/delta
    b1 = 2*Q*(K**2 - 1)/delta
    b2 = ((K**2)*Q - K + Q)/delta
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function BPF_secondorder()

def BSF_secondorder( X, fc, Q, rate=SR ):
    """
    Second order band stop filter

    :param X:
    :param fc:
        corner or resonant freq
    Q
        quality factor
    :param rate:
        sampling rate
    :return:
    """

    K = np.tan( np.pi * fc / rate )
    delta = (K**2)*Q + K + Q
    a0 = Q*(K**2 + 1)/delta
    a1 = 2*Q*(K**2 - 1)/delta
    a2 = Q*(K**2 + 1)/delta
    b1 = 2*Q*(K**2 - 1)/delta
    b2 = ((K**2)*Q - K + Q)/delta
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function BSF_secondorder()


def LPF_secondorder_butterworth( X, fc, rate=SR ):
    """
    Second order Butterworth low pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    :param rate:
        sampling rate
    :return:
    """

    C = 1/(np.tan(np.pi * fc/rate))
    a0 = 1/(1 + np.sqrt(2)*C + C**2)
    a1 = 2*a0
    a2 = a0
    b1 = 2*a0*(1 - C**2)
    b2 = a0*(1 - np.sqrt(2)*C + C**2)
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function LPF_secondorder_butterworth()


def HPF_secondorder_butterworth( X, fc, rate=SR ):
    """
    Second order Butterworth high pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    :param rate:
        sampling rate
    :return:
    """

    C = np.tan(np.pi * fc/rate)
    a0 = 1/(1 + np.sqrt(2)*C + C**2)
    a1 = -2*a0
    a2 = a0
    b1 = -2*a0*(1 - C**2)
    b2 = a0*(1 - np.sqrt(2)*C + C**2)
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function HPF_secondorder_butterworth()


def BPF_secondorder_butterworth( X, fc, BW, rate=SR ):
    """
    Second order Butterworth band pass filter
    doesnt seem to be working ATM

    :param X:
    :param fc:
        corner or cutoff freq
    BW
        bandwidth
    :param rate:
        sampling rate
    :return:
    """

    C = 1/(np.tan(np.pi * fc * BW/rate))
    D = 2*np.cos( 2*np.pi*fc/rate )
    a0 = 1/(1 + C)
    a1 = 0.0
    a2 = -a0
    b1 = -a0*(C*D)
    b2 = a0*(C - 1)
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function BPF_secondorder_butterworth()

def BSF_secondorder_butterworth( X, fc, BW, rate=SR ):
    """
    Second order Butterworth band stop filter

    :param X:
    :param fc:
        corner or cutoff freq
    BW
        bandwidth
    :param rate:
        sampling rate
    :return:
    """

    C = (np.tan(np.pi * fc * BW/rate))
    D = 2*np.cos( 2*np.pi*fc/rate )
    a0 = 1/(1 + C)
    a1 = -a0*D
    a2 = a0
    b1 = -a0*D
    b2 = a0*(1 - C)
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function BSF_secondorder_butterworth()

def APF_firstorder( X, fc, rate=SR ):
    """
    First order all pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    :param rate:
        sampling rate
    :return:
    """

    alpha = (np.tan(np.pi * fc/rate) - 1)/(np.tan(np.pi * fc/rate) + 1)
    a0 = alpha
    a1 = 1.0
    a2 = 0.0
    b1 = alpha
    b2 = 0.0
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function APF_firstorder()

def APF_secondorder( X, fc, Q, rate=SR ):
    """
    Second order all pass filter

    :param X:
    :param fc:
        corner or cutoff freq
    Q
        steepness of phase shift at fc
    :param rate:
        sampling rate
    :return:
    """

    BW = fc/Q
    alpha = (np.tan(BW*np.pi/rate) - 1)/(np.tan(BW*np.pi/rate) + 1)
    beta = -np.cos(2*np.pi*fc/rate)
    a0 = -alpha
    a1 = beta*(1-alpha)
    a2 = 1.0
    b1 = beta*(1-alpha)
    b2 = -alpha
    c0 = 1.0
    d0 = 0.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function APF_secondorder()

def LPShelf_firstorder( X, fc, gain, rate=SR ):
    """
    First order low pass shelf filter

    :param X:
    :param fc:
        corner or cutoff freq
    gain
        gain or attenuation of the shelf, in dB
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2*np.pi*fc/rate
    mu = 10**(  gain/20 )
    beta = 4/(1+mu)
    delta = beta*np.tan(thetac/2)
    gamma = (1-delta)/(1+delta)
    a0 = (1-gamma)/2
    a1 = (1-gamma)/2
    a2 = 0.0
    b1 = -gamma
    b2 = 0.0
    c0 = mu - 1.0
    d0 = 1.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function LPShelf_firstorder()

def HPShelf_firstorder( X, fc, gain, rate=SR ):
    """
    First order high pass shelf filter

    :param X:
    :param fc:
        corner or cutoff freq
    gain
        gain or attenuation of the shelf, in dB
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2*np.pi*fc/rate
    mu = 10**(  gain/20 )
    beta = (1+mu)/4
    delta = beta*np.tan(thetac/2)
    gamma = (1-delta)/(1+delta)
    a0 = (1+gamma)/2
    a1 = -(1+gamma)/2
    a2 = 0.0
    b1 = -gamma
    b2 = 0.0
    c0 = mu - 1.0
    d0 = 1.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function HPShelf_firstorder()

def PEQ_nonconstantQ( X, fc, Q, gain, rate=SR ):
    """
    Second order parametric EQ filter with non constant Q

    :param X:
    :param fc:
        corner or cutoff freq
    Q
        Q
    gain
        gain or attenuation of the shelf, in dB
    :param rate:
        sampling rate
    :return:
    """

    thetac = 2*np.pi*fc/rate
    mu = 10**(  gain/20 )
    zeta = 4/(1+mu)
    beta = 0.5* (1-zeta*np.tan(thetac/(2*Q)))/(1+zeta*np.tan(thetac/(2*Q)))
    gamma = (0.5+beta)*np.cos(thetac)
    a0 = 0.5 - beta
    a1 = 0.0
    a2 = -(0.5-beta)
    b1 = -2*gamma
    b2 = 2*beta
    c0 = mu - 1.0
    d0 = 1.0

    return Biquad_Filter(X, a0, a1, a2, b1, b2, c0, d0)

# end function PEQ_nonconstantQ()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('hi')

    import time

    start_time = time.time()


    # read a wave file
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\zebra.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 08\Clap jmann.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 20\G maj chord.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Desire lines solo crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Major leagues riff crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Forever chords crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\White ferrari chords crop.wav')

    filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filename = 'A string pluck.wav'
    # filename = 'let it live solo.wav'
    filename = 'E4 pluck.wav'
    # filename = 'your dog chords.wav'
    # filename = 'let it live v2.wav'
    # filename = 'desire lines riff2.wav'

    mysong = readWaveFile(filepath+filename)
    t = (1 / SR) * np.arange(0, len(mysong))

    # apply a filter -----------------
    fc  = 1000  # corner freq
    Q   = 1
    BW  = 100
    gain = 30.0

    # mysong = HPF_firstorder(mysong, fc)
    # filtername = 'HPF_1storder' + '_fc' + str(fc)

    # mysong = LPF_firstorder(mysong, fc)
    # filtername = 'LPF_1storder' + '_fc' + str(fc)

    # mysong = LPF_secondorder(mysong, fc, Q)
    # filtername = 'LPF_2ndorder' + '_fc' + str(fc) + '_Q' + st0r(Q)

    mysong = HPF_secondorder(mysong, fc, Q)
    filtername = 'HPF_2ndorder' + '_fc' + str(fc) + '_Q' + str(Q)

    # mysong = BPF_secondorder(mysong, fc, Q)
    # filtername = 'BPF_2ndorder' + '_fc' + str(fc) + '_Q' + str(Q)

    # mysong = BSF_secondorder(mysong, fc, Q)
    # filtername = 'BSF_2ndorder' + '_fc' + str(fc) + '_Q' + str(Q)

    # mysong = LPF_secondorder_butterworth(mysong, fc)
    # filtername = 'LPF_2ndorder_butterworth' + '_fc' + str(fc)

    # mysong = HPF_secondorder_butterworth(mysong, fc)
    # filtername = 'HPF_2ndorder_butterworth' + '_fc' + str(fc)

    # mysong = BPF_secondorder_butterworth(mysong, fc, BW)
    # filtername = 'BPF_2ndorder_butterworth' + '_fc' + str(fc) + '_BW' + str(BW)

    # mysong = APF_firstorder(mysong, fc)
    # filtername = 'APF_1storder' + '_fc' + str(fc)

    # mysong = LPShelf_firstorder(mysong, fc, gain)
    # filtername = 'LPshelf_1storder' + '_fc' + str(fc) + '_gain' + str(gain)

    # mysong = HPShelf_firstorder(mysong, fc, gain)
    # filtername = 'HPshelf_1storder' + '_fc' + str(fc) + '_gain' + str(gain)

    # mysong = PEQ_nonconstantQ(mysong, fc, Q, gain)
    # filtername = 'PEQ' + '_fc' + str(fc) + '_Q' + str(Q) + '_gain' + str(gain)

    print("--- %s seconds ---" % (time.time() - start_time))

    # save
    mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    savefilename = filepath + filename[:-4] + '_' + filtername + '.wav'
    writeWaveFile(savefilename, mysong, rate=SR)