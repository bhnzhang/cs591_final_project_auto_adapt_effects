"""
testing effects
"""

# -------------------------
# Imports
# -------------------------

from f_distortion import *
from f_delay import *
from f_conv_reverb import *
from f_aux_functions import *
from f_filter import *
from f_optimize_effects import *

import pprint

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

    # filepath = 'G:\My Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    filepath = 'D:\Google Drive\Personal projects\py_audio_effects\\2020_06_23_effects\samples\\'
    # filename = 'A string pluck.wav'
    # filename = 'let it live solo.wav'
    # filename = 'E4 pluck.wav'
    filename = 'E2 pluck.wav'
    # filename = 'your dog chords.wav'
    # filename = 'let it live v2.wav'
    # filename = 'desire lines riff2.wav'
    # filename = 'cmaj dmin amin chords.wav'
    # filename = 'Chords2.wav'
    # filename = 'Highriffs.wav'

    mysong = readWaveFile(filepath+filename)
    t = (1 / SR) * np.arange(0, len(mysong))

    # # convolution reverb --------------
    # print('convolution reverb')
    # room_size = 11 # in feet
    # pulse_width = 0.0001
    # decay_time = 0.5
    # duration = 3.0
    # SNR     = 0.5
    # sigma   = 5
    # wetdry = 0.7
    # mysong = Convolution_Reverb(mysong, room_size=room_size,
    #                             pulse_width=pulse_width,
    #                             decay_time=decay_time,
    #                             duration=duration,
    #                             SNR=SNR,
    #                             sigma=sigma,
    #                             wetdry=wetdry )
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_cverb_' + 'roomsize' + str(room_size) + \
    #                '_pulsewid_' + str(pulse_width*1e3) + 'ms_decay_' + str(decay_time) + 's_' + \
    #                 'dur' + str(duration) + 's_SNR' + str(SNR) + \
    #                '_sigma' + str(sigma) + '_wet' + str(wetdry) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # distortion
    # shape   = 'FEXP1'
    # # shape   = 'ATAN'
    # # shape   = 'TANH'
    # # shape   = 'SIG'
    # amount  = 0.5
    # gain    = 0.3
    # print('distortion')
    # mysong = Distortion(mysong, shape, amount, gain)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # # save
    # # mysong = np.array(mysong * (20000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_dist_' + shape + '_amt' + str(amount) + '_gain' + str(gain) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # # # delay --------------
    # delay_time      = 0.1
    # feedback_amt    = 0.3
    # wetdry          = 0.5
    # print('delay')
    # mysong = Delay(X=mysong,
    #                delay_time=delay_time,
    #                feedback_amt=feedback_amt,
    #                wetdry=wetdry,
    #                apply_fb_input=True)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_del_' + 'time' + str(delay_time) + '_fb' \
    #                + str(feedback_amt) + '_wet' + str(wetdry) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # reverb-like delay --------------
    # delay_time = 0.001
    # feedback_amt = 0.9
    # wetdry = 1.0
    # print('reverb-like delay')
    # mysong = Delay(X=mysong,
    #                delay_time=delay_time,
    #                feedback_amt=feedback_amt,
    #                wetdry=wetdry)

    # # reverse reverb? --------------
    # delay_time = np.zeros(len(mysong))
    # delay_time[ t <= np.max(t)/4 ] = 0.01
    # delay_time[ t > np.max(t)/4 ] = 0.1
    # feedback_amt = np.linspace(0.1, 2, len(mysong))
    # feedback_amt[ t >= np.max(t)/2 ] = 0
    # wetdry = 1.0
    # print('reverse reverb-like delay??')
    # mysong = Delay(X=mysong,
    #                delay_time=delay_time,
    #                feedback_amt=feedback_amt,
    #                wetdry=wetdry)

    # # # # trying out chorus ----------
    # moddepth = 0.0005
    # modrate = 10
    # delaytime = 0.01
    # wetdry = 0.7
    # print('chorus')
    # mysong = Chorus(mysong, delaytime, moddepth, modrate, wetdry)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_chor_' + 'dtime' + str(delaytime*1e3) + 'ms_mdep' \
    #                + str(moddepth*1e3) + 'ms_mrate' + str(modrate) + '_wet' + str(wetdry) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)


    # # # # trying out flanger ----------
    # moddepth        = 0.0005
    # modrate         = 2
    # feedback_amt    = 0.7
    # print('flanger')
    # mysong          = Flanger(mysong, moddepth, modrate, feedback_amt)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_flang_' + 'mdep' + str(moddepth*1e3) + \
    #                'ms_mrate' + str(modrate) + '_fb' + str(feedback_amt) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # # trying out vibrato ----------
    # moddepth = 0.0001
    # modrate = 10
    # print('vibrato')
    # mysong = Vibrato(mysong, moddepth, modrate)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # savefilename = filepath + filename[:-4] + '_vib_' + 'mdep' + str(moddepth*1e3) + \
    #                'ms_mrate' + str(modrate) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)


    # # # low pass filter -----------------
    # # LPF design
    # fc = 1000 # corner freq
    # print('LPF')
    # # mysong = Filter(mysong, a0, a1, a2, b1, b2, c0, d0)
    # mysong = LPF_firstorder(mysong, fc)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # # save
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # # savefilename = filepath + filename[:-4] + '_filt_' + \
    # #                 'a0_' + str(a0) + '_a1_' + str(a1) + '_a2_' + str(a2) \
    # #                 + '_b1_' + str(b1) + '_b2_' + str(b2) + '.wav'
    # savefilename = filepath + filename[:-4] + '_filt_LP_fc' + str(fc) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # high pass filter -----------------
    # fc = 10000  # corner freq
    # print('HPF')
    # # mysong = Filter(mysong, a0, a1, a2, b1, b2, c0, d0)
    # mysong = HPF_firstorder(mysong, fc)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # # save
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # # savefilename = filepath + filename[:-4] + '_filt_' + \
    # #                 'a0_' + str(a0) + '_a1_' + str(a1) + '_a2_' + str(a2) \
    # #                 + '_b1_' + str(b1) + '_b2_' + str(b2) + '.wav'
    # savefilename = filepath + filename[:-4] + '_filt_HP_fc' + str(fc) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)

    # # low pass filter 2nd order -----------------
    # fc = 100  # corner freq
    # Q = 0.01
    # print('LPF 2nd order')
    # # mysong = Filter(mysong, a0, a1, a2, b1, b2, c0, d0)
    # mysong = LPF_secondorder(mysong, fc, Q)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # # save
    # mysong = np.array(mysong * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    # # savefilename = filepath + filename[:-4] + '_filt_' + \
    # #                 'a0_' + str(a0) + '_a1_' + str(a1) + '_a2_' + str(a2) \
    # #                 + '_b1_' + str(b1) + '_b2_' + str(b2) + '.wav'
    # savefilename = filepath + filename[:-4] + '_filt_LP2_fc' + str(fc) + '_Q' + str(Q) + '.wav'
    # writeWaveFile(savefilename, mysong, rate=SR)


    # try the first shoegaze effects chain from reverb's youtube vid
    # effects = [
    #     # ('distortion', {'amount': 0.1, 'gain':0.5, 'shape': 'SIG'}),
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 10e3, 'decay_time': 1.0, 'duration': 5.0,
    #                      'SNR': 0.5, 'sigma_f': 50, 'wetdry': 0.5} ),
    #     # ('distortion', {'amount': 0.1, 'gain': 0.5, 'shape': 'TANH'}),
    #     # ('filter', {'filtertype': 'BPF_2ndorder', 'fc': 1e3, 'Q': 0.5}),
    #     ('tremolo', {'depth': 0.3, 'modrate': 10})
    # ]
    # songname = filename[:-4] + '_shoegaze1_nodistort.wav'
    # effects = [
    #     ('distortion', {'amount': 0.3, 'gain':0.5, 'shape': 'SIG'}),
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 10e3, 'decay_time': 1.0, 'duration': 5.0,
    #                      'SNR': 0.5, 'sigma_f': 50, 'wetdry': 0.5}),
    #     # ('distortion', {'amount': 0.1, 'gain': 0.5, 'shape': 'TANH'}),
    #     # ('filter', {'filtertype': 'BPF_2ndorder', 'fc': 1e3, 'Q': 0.5}),
    #     ('tremolo', {'depth': 0.3, 'modrate': 10})
    # ]
    # songname = filename[:-4] + '_shoegaze1_distort_b4_reverb.wav'
    # effects = [
    #     ('distortion', {'amount': 0.1, 'gain': 0.5, 'shape': 'SIG'}),
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 10e3, 'decay_time': 1.0, 'duration': 5.0,
    #                      'SNR': 0.5, 'sigma_f': 50, 'wetdry': 0.5}),
    #     ('distortion', {'amount': 0.1, 'gain': 0.5, 'shape': 'FEXP1'}),
    #     ('filter', {'filtertype': 'LPF_2ndorder', 'fc': 10e3, 'Q': 0.5}),
    #     ('tremolo', {'depth': 0.3, 'modrate': 10})
    # ]
    # songname = filename[:-4] + '_shoegaze1_distort.wav'
    # effects = [
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 10e3, 'decay_time': 1.0, 'duration': 2.0,
    #                      'SNR': 0.5, 'sigma_f': 50, 'wetdry': 0.5} ),
    #     ('tremolo', {'depth': 0.3, 'modrate': 0.3})
    # ]
    # songname = filename[:-4] + '_revtrem.wav'

    # # pure delay effects
    # effects = [
    #     ('delay', {'delay_time': 0.2, 'feedback_amt': 0.6, 'wetdry': 0.4}),
    #     ('chorus', {'delay_time': 0.01, 'moddepth': 0.3, 'modrate': 0.4, 'wetdry': 0.3})
    # ]
    # songname = filename[:-4] + '_delaychorus.wav'
    # effects = [
    #     ('delay', {'delay_time': 0.4, 'feedback_amt': 0.6, 'wetdry': 0.4}),
    #     ('chorus', {'delay_time': 0.01, 'moddepth': 0.3, 'modrate': 0.4, 'wetdry': 0.3}),
    #     ('flanger', {'moddepth': 0.1, 'modrate': 0.2, 'feedback_amt': 0.5})
    # ]
    # effects = [
    #     ('delay', {'delay_time': 0.4, 'feedback_amt': 0.6, 'wetdry': 0.4}),
    #     ('chorus', {'delay_time': 0.01, 'moddepth': 0.3, 'modrate': 0.4, 'wetdry': 0.3}),
    #     ('distortion', {'amount': 0.05, 'gain': 0.5, 'shape': 'SIG'})
    # ]
    # songname = filename[:-4] + '_delaychorusOD.wav'
    # effects = [
    #     ('chorus', {'delay_time': 0.01, 'moddepth': 0.4, 'modrate': 0.5, 'wetdry': 0.4}),
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 3e3, 'decay_time': 0.5, 'duration': 10.0,
    #                           'SNR': 0.0, 'sigma_f': 50, 'wetdry': 0.4}),
    #     ('filter', {'filtertype': 'BPF_2ndorder', 'fc': 500, 'Q': 0.5}),
    # ]
    # songname = filename[:-4] + '_chorusreverbBPF.wav'
    # effects = [
    #     ('chorus', {'delay_time': 0.01, 'moddepth': 0.4, 'modrate': 0.5, 'wetdry': 0.4}),
    #     ('conv_reverb', {'room_size': 20, 'cutoff_freq': 3e3, 'decay_time': 0.5, 'duration': 10.0,
    #                      'SNR': 0.0, 'sigma_f': 50, 'wetdry': 0.4}),
    # ]
    # songname = filename[:-4] + '_chorusreverb.wav'
    effects = [
        ('delay', {'delay_time': 0.2, 'feedback_amt': 0.9, 'wetdry': 0.4}),
        ('chorus', {'delay_time': 0.01, 'moddepth': 0.3, 'modrate': 0.4, 'wetdry': 0.3}),
        ('vibrato', {'moddepth': 0.8, 'modrate': 0.01})
    ]
    songname = filename[:-4] + '_delaychorusvib.wav'

    # effects = [
    #     ('distortion', {'shape': 'FEXP1', 'amount': 0.9, 'gain': 0.2}),
    #     ('filter', {'filtertype': 'HPShelf_1storder', 'fc': 0.5, 'gain': -10})
    # ]
    # songname = filename[:-4] + '_distort_filt.wav'

    mysong2 = Effects_Chain(mysong, effects)
    print("--- %s seconds ---" % (time.time() - start_time))

    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(effects)

    # # save
    mysong2 = np.array(mysong2 * (25000 / np.amax(mysong))).astype(np.int16)  # rescale sound
    writeWaveFile(filepath + songname, mysong2, rate=SR)

    print('correlation between original and mod''ed song:')
    print(compare_signal_similarity(mysong, mysong2))
    print("--- %s seconds ---" % (time.time() - start_time))