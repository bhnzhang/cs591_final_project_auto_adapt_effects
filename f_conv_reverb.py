"""
Convolution reverb
"""

# -------------------------
# Imports
# -------------------------

# numpy
import numpy as np

# scipy signal
import scipy.signal as sig

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

def gaussian_comb_IR(room_size, cutoff_freq, decay_time, duration, SNR, sigma_f, rate=SR):
    """
    Makes a gaussian comb impulse response for convolution reverb
    The resulting IR is a train of gaussian pulses, of a given repetition time, pulse widths and decay time

    it's probably recommended that the duration = integer multiple of rep_time

    inputs are:
        room_size
            Width of a cubic room in feet
        cutoff_freq
            1/e cutoff frequency
        decay_time
            1/e decay time of the time envelope, in seconds
        duration
            total duration time of the IR
    """

    vsound = 1125 # speed of sound, ft/s

    # get number of samples
    n_samps = int(round(duration*rate))

    t = np.arange(n_samps) / rate  # time vector, in seconds
    f = np.fft.fftfreq(n_samps, 1 / SR)  # freq vector

    # # make gaussian pulse
    # tpulse  = t - t[ int(round(len(t)/2)) ]
    # pulse   = np.exp( -(1/2) * ((tpulse/pulse_width)**2) )
    # pulse   = np.fft.fftshift(pulse) # i think the pulse needs to be shifted to time 0

    # # make impulse train, old version
    # delta_positions = np.array(np.arange(0, n_samps-1, int(round(rep_time*SR))))
    # impulse_train = signal.unit_impulse(n_samps, delta_positions)

    # make impulse train in freq domain
    N_pos               = int(math.ceil(n_samps/2.0))
    x                   = np.arange(N_pos)
    x                   = x/N_pos
    y                   = ( (N_pos - 1)* (1 - (x ** 2)) ).astype(int)
    y = y[::-1]

    # plt.figure()
    # plt.plot(y)
    # plt.show()

    fund_freq         = int((vsound/2) * (1.0/room_size) * (N_pos/rate))
    delta_pos_posfreq   = y[ fund_freq:N_pos-1:fund_freq ]
    delta_pos_posfreq   = delta_pos_posfreq[ delta_pos_posfreq < N_pos ]
    # delta_pos_posfreq   = np.arange( fund_freq, N_pos-1, fund_freq )
    impulse_train       = signal.unit_impulse(N_pos, delta_pos_posfreq)

    # plt.figure()
    # plt.plot(f[0:len(impulse_train)],impulse_train)
    # plt.xlabel('frequency')
    # plt.show()

    # make random comb
    df = rate/N_pos
    sigma = int(round(sigma_f/df))
    delta_pos_posfreq_random = delta_pos_posfreq + np.random.randint(low=-sigma,
                                                                     high=sigma,
                                                                     size=len(delta_pos_posfreq))
    delta_pos_posfreq_random = delta_pos_posfreq_random[ delta_pos_posfreq_random > 0 ]
    delta_pos_posfreq_random = delta_pos_posfreq_random[delta_pos_posfreq_random < N_pos]
    impulse_train_noise = signal.unit_impulse(N_pos, delta_pos_posfreq_random)

    # combine clean and noise reverb IRs
    impulse_train = SNR * impulse_train + (1 - SNR) * impulse_train_noise

    # cut off high frequencies
    impulse_train = impulse_train * np.exp( -f[0:len(impulse_train)]/cutoff_freq )

    # combine positive and negative frequencies
    if n_samps % 2 is 0:
        impulse_train = np.hstack( [ impulse_train, impulse_train[::-1] ] )
    else:
        impulse_train = np.hstack( [impulse_train, impulse_train[-1:0:-1]] )

    # plt.figure()
    # plt.plot(impulse_train)
    # plt.xlabel('frequency')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(t, np.fft.ifft(impulse_train))
    # plt.xlabel('time')
    # plt.show()

    # apply gaussian envelope
    # ir = np.fft.ifft( np.fft.fft(impulse_train) * np.fft.fft(pulse) )
    # ir = np.fft.ifft(impulse_train * np.fft.fft(pulse))
    ir = np.fft.ifft(impulse_train)

    # plt.figure()
    # plt.plot(ir)
    # plt.show()

    # plt.figure()
    # plt.plot(f,np.fft.fft(pulse))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(f,np.fft.fft(ir))
    # plt.show()

    #
    # plt.figure()
    # plt.plot(np.fft.fft(pulse))
    # plt.show()

    # multiply by decay
    ir = ir * np.exp( -t/decay_time )

    # remove initial ping
    ir[t < 0.001] = 0

    # re-scale power to equal to 1
    ir = ir /np.sqrt( np.sum( np.abs(ir) **2 ) )



    # # # # debug for viewing
    # plt.figure()
    # plt.plot(t, ir)
    # plt.show()
    # #
    # plt.figure()
    # plt.plot(np.fft.fft(ir))
    # plt.show()

    return ir

# end function gaussian_comb_IR()


def Convolution_Reverb( X, room_size, cutoff_freq, decay_time, duration, SNR, sigma_f, wetdry, rate=SR ):
    """
    Implements convolution reverb
    room_size, pulse_width, decay_time can be iterables
    duration cannot be.

    :param X:
    :param room_size:
    :param pulse_width:
    :param decay_time:
    :param duration:
    :param rate:
    :return:
    """

    if not is_iterable(room_size):
        ir = gaussian_comb_IR(room_size, cutoff_freq, decay_time, duration, SNR, sigma_f)
    else:
        ir = gaussian_comb_IR(room_size[0], cutoff_freq[0], decay_time[0], duration, SNR, sigma_f)
        for r,p,de in zip(room_size[1:], cutoff_freq[1:], decay_time[1:]):
            ir += gaussian_comb_IR(r,p,de,duration,SNR, sigma_f)
        ir = ir / np.sum(np.abs(ir) ** 2)  # re-scale power to equal to 1


    # zero pad if necessary
    if len(ir) > len(X):
        X = np.hstack( (X, np.zeros( len(ir)- len(X) )) )
    elif len(X) > len(ir):
        ir = np.hstack( (ir, np.zeros( len(X)- len(ir) )) )

    # convolve
    Y = np.fft.ifft( np.fft.fft(ir) * np.fft.fft(X) )

    # return wetdry
    return Y*wetdry + X*(1-wetdry)

# end Convolution_Reverb

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('f_conv_reverb.py')

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
    # t = (1 / SR) * np.arange(0, len(mysong))

    filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filename = 'A string pluck.wav'
    # filename = 'let it live solo.wav'
    filename = 'E4 pluck.wav'
    # filename = 'your dog chords.wav'
    # filename = 'let it live v2.wav'

    mysong = readWaveFile(filepath + filename)
    t = (1 / SR) * np.arange(0, len(mysong))

    # # # convolution reverb
    # rep_time    = 0.1
    # pulse_width = 0.0001
    # decay_time  = 5.0
    # duration    = 20.0
    # wetdry      = 0.9
    # mysong = Convolution_Reverb( mysong, rep_time, pulse_width, decay_time, duration, wetdry )

    # # convolution reverb
    # rep_time = [ 0.05, 0.01]
    # pulse_width =[ 0.0005, 0.0005]
    # decay_time = [ 0.5, 2.0 ]
    # duration = 20.0
    # wetdry = 0.0
    # mysong = Convolution_Reverb(mysong, rep_time, pulse_width, decay_time, duration, wetdry)

    # # convolution "reverse" reverb
    # rep_time = [ 0.05, 0.01 ]
    # pulse_width = [ 0.0003, 0.0003]
    # decay_time = [ -0.5, 3.0 ]
    # duration = 2.0
    # wetdry = 1.0
    # mysong = Convolution_Reverb(mysong, rep_time, pulse_width, decay_time, duration, wetdry)
    #
    # # save
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # writeWaveFile('test.wav', mysong, rate=SR)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))

    # convolution reverb --------------
    print('convolution reverb')
    room_size   = 40 # in feet
    cutoff_freq = 3000
    decay_time  = 0.3
    duration    = 2.0
    SNR         = 0.5
    sigma_f     = 100
    wetdry      = 0.3
    mysong = Convolution_Reverb(mysong, room_size=room_size,
                                cutoff_freq=cutoff_freq,
                                decay_time=decay_time,
                                duration=duration,
                                SNR=SNR,
                                sigma_f=sigma_f,
                                wetdry=wetdry )

    print("--- %s seconds ---" % (time.time() - start_time))
    mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    savefilename = filepath + filename[:-4] + '_cverb_' + 'roomsize' + str(room_size) + \
                   '_freqcut_' + str(cutoff_freq) + '_decay_' + str(decay_time) + 's_' + \
                    'dur' + str(duration) + 's_SNR' + str(SNR) + \
                   '_sigma' + str(sigma_f) + '_wet' + str(wetdry) + '.wav'
    writeWaveFile(savefilename, mysong, rate=SR)


    # # # load and plot IR and TF of a room
    # roomIR = readWaveFile(filepath + 'HepnerHall.wav')

    # plt.figure()
    # plt.plot(roomIR)
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.fft.fftfreq(len(roomIR), 1/SR), np.fft.fft(roomIR))
    # plt.show()

