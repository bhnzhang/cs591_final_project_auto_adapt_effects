"""
Delay functions and modulated delays
"""

# -------------------------
# Imports
# -------------------------

# numpy
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

def Delay( X, delay_time, feedback_amt, wetdry, apply_fb_input=True, rate=SR ):
    """
    Delay audio via circular buffer

    :param X:
        type: numpy array
        desc: input, dry signal
    :param delay_time:
        type: float, scalar or vec
        desc: amount of delay time, in seconds
    :param feedback_amt:
        type: float, scalar or vec
        desc: amount of feedback, ranging from 0 to 1, inclusive
    :param wetdry:
        type: float, scalar or vec
        desc: amount of wet to dry mix, ranging from 0 to 1, inclusive
                0 = fully dry
                1 = fully wet
    :param apply_fb_input:
        type: bool
        desc: if true, delay buffer write value = (X+Y)*feedback
              if false, delay buffer write value = X + Y*feedback
    :return:
        delayed audio data
    """

    # convert inputs if scalars into np arrays
    delay_time      = delay_time * np.ones(len(X)) if np.isscalar(delay_time) else delay_time
    feedback_amt    = feedback_amt * np.ones(len(X)) if np.isscalar(feedback_amt) else feedback_amt
    wetdry          = wetdry * np.ones(len(X)) if np.isscalar(wetdry) else wetdry

    # convert delay time to delay in samples
    # not implemented yet, but eventually would be good to interpolate
    maxdelay    = np.max(delay_time)
    # delay_samps = np.array(delay_time*SR).astype(int)
    delay_samps = delay_time * SR

    # create circular buffer with appropriate size
    buffer_size = int( math.ceil( math.log(maxdelay*rate,2) ) )
    # print(buffer_size)
    delay_size  = int(round(maxdelay*SR)) # approximate for now
    delaybuff   = Circ_buffer( buffer_size=buffer_size, delay_size=delay_size )

    # make output vec
    output_sig = np.zeros(len(X)).astype(int)

    # process signal
    for ii in range(len(X)):

        # read delayed value
        delay_prev_samp = math.ceil( delay_samps[ii] )
        prev_samp       = delaybuff.read_value( delay_prev_samp )
        delay_next_samp = delay_prev_samp - 1
        next_samp       = delaybuff.read_value(delay_next_samp)
        output_sig[ii]  = prev_samp + ( (next_samp - prev_samp) * (delay_prev_samp - delay_samps[ii]) )

        # calculate value to write
        if apply_fb_input:
            cur_value = (X[ii] + output_sig[ii]) * feedback_amt[ii]
        else:
            cur_value = X[ii] + output_sig[ii]*feedback_amt[ii]

        # write to buffer
        delaybuff.write_value(int(cur_value))

    # end for loop

    # return output
    return np.array(output_sig * wetdry + X * (1-wetdry)).astype(np.int16)
# end function Delay()


def Flanger(X, moddepth, modrate, feedback_amt, rate=SR ):
    """
    Flanger effect

    typical settings:
    set moddepth to maximum of 2-7 ms
    modrate between sub Hz to 10 Hz
    feedback, you can go wild with

    :param X:
        type: int16, vector
        desc: audio signal
    :param moddepth:
        type: float, scalar
        desc: max amount of delay modulation, in seconds
    :param modrate:
        type: float, scalar
        desc: modulation rate, in Hz
    :param feedback_amt:
        type: float, scalar
        desc: feedback amount, between -1 to 1
    :param rate:
        type: int
        desc: sampling rate
    :return:
    """
    LFO = make_triangle_LFO(len(X), modrate/rate, moddepth) + moddepth + 1/rate
    return Delay(X, LFO, feedback_amt, 0.5, rate=rate )
# end function Flanger()

def Vibrato(X, moddepth, modrate, rate=SR ):
    """
    Vibrato effect

    typical settings:
    set moddepth to maximum of 2-7 ms
    modrate between sub Hz to 10 Hz

    :param X:
        type: int16, vector
        desc: audio signal
    :param moddepth:
        type: float, scalar
        desc: max amount of delay modulation, in seconds
    :param modrate:
        type: float, scalar
        desc: modulation rate, in Hz
    :param rate:
        type: int
        desc: sampling rate
    :return:
    """
    LFO = make_sin_LFO(len(X), modrate / rate, moddepth) + moddepth + 1 / rate
    return Delay(X, LFO, 0, 1, False, rate=rate )
# end function Vibrato()

def Chorus(X, delay_time, moddepth, modrate, wetdry, rate=SR):
    """
    Chorus effect

    for typical chorus, set the delay time to something on order of 20-50ms
    set moddepth to up to 2-7 ms, like flanger and vibrato

    :param X:
    :param delay_time:
    :param moddepth:
    :param modrate:
    :param wetdry:
    :param rate:
    :return:
    """
    LFO = make_sin_LFO(len(X), modrate / rate, moddepth) + delay_time
    return Delay(X, LFO, 0, wetdry, False, rate=rate)
# end function Chorus()

def Delay_distortion( X, delay_time, feedback_amt, wetdry, distortion_amt, rate=SR ):
    """
    Delay audio via circular buffer
    with distortion built in

    :param X:
        type: numpy array
        desc: input, dry signal
    :param delay_time:
        type: float, scalar or vec
        desc: amount of delay time
    :param feedback_amt:
        type: float, scalar or vec
        desc: amount of feedback, ranging from 0 to 1, inclusive
    :param wetdry:
        type: float, scalar or vec
        desc: amount of wet to dry mix, ranging from 0 to 1, inclusive
                0 = fully dry
                1 = fully wet
    :return:
        delayed audio data
    """

    # convert inputs if scalars into np arrays
    delay_time      = delay_time * np.ones(len(X)) if np.isscalar(delay_time) else delay_time
    feedback_amt    = feedback_amt * np.ones(len(X)) if np.isscalar(feedback_amt) else feedback_amt
    wetdry          = wetdry * np.ones(len(X)) if np.isscalar(wetdry) else wetdry

    # convert delay time to delay in samples
    maxdelay    = np.max(delay_time)
    delay_samps = np.array(delay_time*SR).astype(int)

    # create circular buffer with appropriate size
    buffer_size = int( math.ceil( math.log(maxdelay*rate,2) ) )
    # print(buffer_size)
    delay_size  = int(round(maxdelay*SR)) # approximate for now
    delaybuff   = Circ_buffer( buffer_size=buffer_size, delay_size=delay_size )

    # make output vec
    output_sig = np.zeros(len(X)).astype(int)

    # process signal
    for ii in range(len(X)):

        # set delay time
        delaybuff.set_delay( delay_samps[ii] )

        # read delayed value
        output_sig[ii] = delaybuff.read_value()

        # calculate value to write
        cur_value = X[ii] + output_sig[ii]*feedback_amt[ii]

        # add distortion
        cur_value = Distortion(cur_value, 'SIG', distortion_amt)

        # write to buffer
        delaybuff.write_value(int(cur_value))

    # end for loop

    # return output
    return np.array(output_sig * wetdry + X * (1-wetdry)).astype(np.int16)

# end function Delay()

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print('hi')

    # sig = np.zeros(100)
    # output_sig = Delay(X = sig, delay_time=2, feedback_amt=0, wetdry=1)

    # read a wave file
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\zebra.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 08\Clap jmann.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 20\G maj chord.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Desire lines solo crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Major leagues riff crop.wav')
    # mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\Forever chords crop.wav')
    mysong = readWaveFile('G:\My Drive\Personal projects\iphone recordings\\2020 06 21\White ferrari chords crop.wav')
    t = (1 / SR) * np.arange(0, len(mysong))


    # trying out delay --------------
    # delay_time = 0.5
    # feedback_amt = 0.7
    # wetdry = 0.5
    # print('delay')
    # mysong = Delay( X = mysong,
    #                       delay_time = delay_time,
    #                       feedback_amt = feedback_amt,
    #                       wetdry = wetdry)

    # # # trying out flanger ----------
    # moddepth        = 0.0005
    # modrate         = 0.5
    # feedback_amt    = 0.2
    # print('flanger')
    # mysong          = Flanger(mysong, moddepth, modrate, feedback_amt)

    # # # trying out vibrato ----------
    # moddepth = 0.0001
    # modrate = 5
    # print('vibrato')
    # mysong = Vibrato(mysong, moddepth, modrate)

    # # # # trying out chorus ----------
    # moddepth    = 0.0001
    # modrate     = 5
    # delaytime   = 0.02
    # wetdry      = 0.5
    # print('chorus')
    # mysong = Chorus(mysong, delaytime, moddepth, modrate, wetdry)

    # trying out delay with distortion in feedback loop --------------
    delay_time      = 0.5
    feedback_amt    = 0.7
    wetdry          = 0.5
    distort_amt     = 10
    print('delay distortion')
    mysong = Delay_distortion( X = mysong,
                          delay_time = delay_time,
                          feedback_amt = feedback_amt,
                          wetdry = wetdry,
                               distortion_amt= distort_amt)

    # save
    mysong = np.array(mysong * (25000/np.amax(mysong))).astype(np.int16) # rescale sound
    writeWaveFile('test.wav', mysong, rate=SR)