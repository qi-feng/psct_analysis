try:
    import target_io
    import target_driver
except:
    print("Cannot import target libraries")

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

import datetime
import time

import numpy as np
import scipy.io
import scipy as sp
from scipy.optimize import curve_fit
from scipy.signal import medfilt2d

#import h5py
import pickle
import argparse


# some global var that can be modified to a config file
#DATADIR='/a/data/tehanu/pSCTdata'
DATADIR='/mnt/data476G/pSCT_data/'

#OUTDIR='/a/data/tehanu/qifeng/pSCT/results/'
OUTDIR='./'

numBlock = 4
nSamples = 32*numBlock
nchannel = 16
nasic = 4
chPerPacket = 32

modList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
nModules = len(modList)

modPos = {4: 5, 5: 6, 1: 7, 3: 8, 2: 9,
              103: 11, 125: 12, 126: 13, 106: 14, 9: 15,
              119: 17, 108: 18, 110: 19, 121: 20, 8: 21,
              115: 23, 123: 24, 124: 25, 112: 26, 7: 27,
              100: 28, 111: 29, 114: 30, 107: 31, 6: 32,
              101: 14}  # 101 was formerly in slot 14 before it broke

posGrid = {5: (1, 1), 6: (1, 2), 7: (1, 3), 8: (1, 4), 9: (1, 5),
               11: (2, 1), 12: (2, 2), 13: (2, 3), 14: (2, 4), 15: (2, 5),
               17: (3, 1), 18: (3, 2), 19: (3, 3), 20: (3, 4), 21: (3, 5),
               23: (4, 1), 24: (4, 2), 25: (4, 3), 26: (4, 4), 27: (4, 5),
               28: (5, 1), 29: (5, 2), 30: (5, 3), 31: (5, 4), 32: (5, 5)}


def row_col_coords(index):
    # Convert bits 1, 3 and 5 to row
    row = 4*((index & 0b100000) > 0) + 2*((index & 0b1000) > 0) + 1*((index & 0b10) > 0)
    # Convert bits 0, 2 and 4 to col
    col = 4*((index & 0b10000) > 0) + 2*((index & 0b100) > 0) + 1*((index & 0b1) > 0)
    return (row, col)

# calculating the actual index reassignments
row, col = row_col_coords(np.arange(64))




# io

def row_col_coords(index):
    # get position for pixels within a module
    # Convert bits 1, 3 and 5 to row
    row = 4 * ((index & 0b100000) > 0) + 2 * ((index & 0b1000) > 0) + 1 * ((index & 0b10) > 0)
    # Convert bits 0, 2 and 4 to col
    col = 4 * ((index & 0b10000) > 0) + 2 * ((index & 0b100) > 0) + 1 * ((index & 0b1) > 0)
    return (row, col)


def calcLoc(modInd, full=True):
    # determine the location of the module in the gridspace
    if full:
        reflectList = [4, 3, 2, 1, 0]
        loc = tuple(np.subtract(posGrid[modPos[modList[modInd]]], (1, 1)))
        locReflect = tuple([loc[0], reflectList[loc[1]]])
        return loc, locReflect
    else:
        reflectList = [3, 2, 1, 0]
        loc = tuple(np.subtract(posGrid[modPos[modList[modInd]]], (2, 1)))
        locReflect = tuple([loc[0], reflectList[loc[1]]])
        return loc, locReflect


def get_reader(run_num, numBlock=4, nchannel=16,
               nasic=4, chPerPacket=32,
               modList=[1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125,
                        126],
               DATADIR=DATADIR,
               OUTDIR=OUTDIR,
               maxZ=1000

               ):

    filename = "{}/run{}.fits".format(DATADIR, run_num)
    reader = target_io.EventFileReader(filename)
    nEvents = reader.GetNEvents()
    print("number of events in file {}: {}".format(filename, nEvents))
    return reader


def get_reader_calibrated(filename,
               DATADIR=DATADIR,
               ):

    filename = "{}/{}".format(DATADIR, filename)
    reader = target_io.EventFileReader(filename)
    nEvents = reader.GetNEvents()
    print("number of events in file {}: {}".format(filename, nEvents))
    return reader

def event_reader_evtloop_first(reader, event_list=range(10)):
    for ievt in event_list:
        for modInd in range(len(modList)):
            for asic in range(nasic):
                for ch in range(nchannel):
                    rawdata = reader.GetEventPacket(ievt, int((((nasic * modInd + asic) * nchannel) + ch) / chPerPacket))
                    packet = target_driver.DataPacket()
                    packet.Assign(rawdata, reader.GetPacketSize())
                    header = target_driver.EventHeader()
                    reader.GetEventHeader(ievt, header)
                    blockNumber = (packet.GetRow() + packet.GetColumn() * 8)
                    blockPhase = (packet.GetBlockPhase())
                    timestamp = packet.GetTACKTime()
                    wf = packet.GetWaveform((asic * nchannel + ch) % chPerPacket)
                    for sample in range(nSamples):
                        # ampl[ievt, modInd, asic, ch, sample] = wf.GetADC(sample)
                        yield ievt, modInd, asic, ch, sample, wf.GetADC(sample), blockNumber, blockPhase, timestamp


def event_reader(reader, event_list=range(10)):
    for modInd in range(len(modList)):
        for asic in range(nasic):
            for ch in range(nchannel):
                for ievt in event_list:
                    rawdata = reader.GetEventPacket(ievt, int((((nasic * modInd + asic) * nchannel) + ch) / chPerPacket))
                    packet = target_driver.DataPacket()
                    packet.Assign(rawdata, reader.GetPacketSize())
                    header = target_driver.EventHeader()
                    reader.GetEventHeader(ievt, header)
                    blockNumber = (packet.GetRow() + packet.GetColumn() * 8)
                    blockPhase = (packet.GetBlockPhase())
                    timestamp = packet.GetTACKTime()
                    wf = packet.GetWaveform((asic * nchannel + ch) % chPerPacket)
                    for sample in range(nSamples):
                        # ampl[ievt, modInd, asic, ch, sample] = wf.GetADC(sample)
                        yield ievt, modInd, asic, ch, sample, wf.GetADC(sample), blockNumber, blockPhase, timestamp


def get_trace(ampl, ievt, modInd, asic, ch, show=False, ax=None, title=None, ylim=None):
    if show:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(ampl[ievt, modInd, asic, ch, :])
        ax.set_xlabel("Sample")
        ax.set_ylabel("ADC")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
    return ampl[ievt, modInd, asic, ch, :]


def get_trace_window(ampl, ievt, modInd, asic, ch,
                     frac=0.1, verbose=True,
                     show=False, ax=None, title=None):
    if show:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        tr_ = ampl[ievt, modInd, asic, ch, :]
        tr_peak = np.max(tr_)
        sam_peak = int(np.where(tr_==tr_peak)[0])
        print(sam_peak)
        sams = np.arange(1, 1+nSamples)
        tr_baseline = (np.median(tr_[:15]) + np.median(tr_[-30:])) / 2.
        tr_startADC = tr_baseline + frac*(tr_peak-tr_baseline)
        start_sample = int(np.interp(tr_startADC,  tr_[sam_peak-20:sam_peak], sams[sam_peak-20:sam_peak]))
        stop_sample = int(np.interp(tr_startADC, tr_[sam_peak:sam_peak+20][::-1],  sams[sam_peak:sam_peak+20][::-1]))
        #print(tr_baseline, tr_startADC, tr_peak, sam_peak, start_sample, stop_sample)
        int_samples = stop_sample - start_sample
        mean_c = np.mean(tr_[start_sample:stop_sample+1]-tr_baseline)
        if verbose:
            print("Integration window {}, mean charge {:.1f} ADC (subtract baseline {:.1f} ADC)".format(int_samples, mean_c, tr_baseline))
        if show:
            ax.plot(sams, tr_-tr_baseline)
            ax.axvline(start_sample, ls="--", alpha=0.5)
            ax.axvline(stop_sample, ls="--", alpha=0.5)
            ax.set_xlabel("Sample")
            ax.set_ylabel("ADC [subtract basline]")
            ax.set_title(title)
    return ampl[ievt, modInd, asic, ch, :]


def get_trace_window_block_test(ampl, ievt, modInd, asic, ch,
                                frac=0.1, verbose=True,
                                blocks=None, phases=None,
                                show=False, ax=None, title=None,
                                ylim=None):
    if show:
        sam_b1 = -1
        if blocks is not None:
            b_ = blocks[ievt]
        if phases is not None:
            ph_ = int(phases[ievt])
            sam_b1 = 32 - ph_
        else:
            print("No phases provided, why am I called")
            return -1, -1
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        tr_ = ampl[ievt, modInd, asic, ch, :]
        tr_peak = np.max(tr_[25:90])
        if np.sum(tr_) == 0:
            #empty trace
            return ampl[ievt, modInd, asic, ch, :], 0, 0, 0, 0
        peak_ind = np.where(tr_ == tr_peak)
        #print(peak_ind)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        if len(peak_ind[0])>1:
            #print("more than 1")
            peak_ind = find_nearest(peak_ind[0], 45)
        else:
            peak_ind = peak_ind[0]
        #print(peak_ind)

        sam_peak = int(peak_ind)
        #print(sam_peak)
        sams = np.arange(nSamples)

        # first block and last block are probably safe for pedestal estimation
        tr_baseline_b1 = np.median(tr_[:sam_b1])
        tr_[:sam_b1] = tr_[:sam_b1] - tr_baseline_b1

        tr_baseline_b5 = np.median(tr_[sam_b1 + 96:])
        tr_[sam_b1 + 96:] = tr_[sam_b1 + 96:] - tr_baseline_b5

        # tr_baseline = np.mean(tr_baseline_b1, tr_baseline_b2, tr_baseline_b3, tr_baseline_b4, tr_baseline_b5)
        """
        if sam_b1<15: 
            # in this case the second block is also good for pedestal
            tr_baseline_b2 = np.median(tr_[sam_b1:sam_b1+32]) 
            tr_[sam_b1:sam_b1+32] = tr_[sam_b1:sam_b1+32]-tr_baseline_b2
            tr_baseline = np.mean([tr_baseline_b1, tr_baseline_b2, tr_baseline_b5])      
        else:
            tr_baseline = np.mean([tr_baseline_b1, tr_baseline_b5])    
        """
        tr_baseline = np.mean([tr_baseline_b1, tr_baseline_b5])

        tr_startADC = tr_baseline + frac * (tr_peak - tr_baseline)
        print(tr_baseline, tr_startADC, tr_peak, sam_peak)

        if (tr_peak-tr_baseline)<48:
            start_sample=40
            stop_sample=80
        else:

            start_sample = int(np.interp(tr_startADC, tr_[max(0, sam_peak - 20):sam_peak], sams[max(0, sam_peak - 20):sam_peak]))
            stop_sample = int(np.interp(tr_startADC, tr_[sam_peak:min(sam_peak + 20, nSamples)][::-1], sams[sam_peak:min(sam_peak + 20, nSamples)][::-1]))
        # print(tr_baseline, tr_startADC, tr_peak, sam_peak, start_sample, stop_sample)

        int_samples = stop_sample - start_sample

        # if sam_b1 >=15:
        if True:
            if (sam_b1 + 32) < start_sample or sam_b1 > stop_sample:
                tr_baseline_b2 = np.median(tr_[sam_b1:sam_b1 + 32])
            elif sam_b1 > start_sample and (sam_b1 + 32) < stop_sample:
                # fully contained... use the previous block
                tr_baseline_b2 = tr_baseline
            elif sam_b1 > start_sample and (sam_b1 + 32) > stop_sample:
                # tr_baseline_b2 = np.median(tr_[stop_sample:sam_b1+32])
                tr_baseline_b2 = tr_baseline
            elif sam_b1 < start_sample and (sam_b1 + 32) > stop_sample:
                # both sides can be used
                # tr_baseline_b2 = np.median([tr_[sam_b1:start_sample],tr_[stop_sample:sam_b1+32]])
                tr_baseline_b2 = tr_baseline
            elif sam_b1 < start_sample and (sam_b1 + 32) < stop_sample:
                # tr_baseline_b2 = np.median(tr_[sam_b1:start_sample])
                tr_baseline_b2 = tr_baseline
            else:
                print("Shouldn't reach here")
                # tr_baseline_b2 = tr_baseline_b1
                tr_baseline_b2 = tr_baseline

        tr_[sam_b1:sam_b1 + 32] = tr_[sam_b1:sam_b1 + 32] - tr_baseline_b2

        if (sam_b1 + 64) < start_sample or (sam_b1 + 32) > stop_sample:
            tr_baseline_b3 = np.median(tr_[sam_b1 + 32:sam_b1 + 64])
        elif (sam_b1 + 32) > start_sample and (sam_b1 + 64) < stop_sample:
            # fully contained... use the average block
            tr_baseline_b3 = tr_baseline
        elif (sam_b1 + 32) > start_sample and (sam_b1 + 64) > stop_sample:
            # tr_baseline_b3 = np.median(tr_[stop_sample:sam_b1+64])
            tr_baseline_b3 = tr_baseline
        elif (sam_b1 + 32) < start_sample and (sam_b1 + 64) > stop_sample:
            # tr_baseline_b3 = np.median([tr_[sam_b1+32:start_sample],tr_[stop_sample:sam_b1+64]])
            tr_baseline_b3 = tr_baseline
        elif (sam_b1 + 32) < start_sample and (sam_b1 + 64) < stop_sample:
            # tr_baseline_b3 = np.median(tr_[sam_b1+32:start_sample])
            tr_baseline_b3 = tr_baseline
        else:
            print("Shouldn't reach here")
            tr_baseline_b3 = tr_baseline

        tr_[sam_b1 + 32:sam_b1 + 64] = tr_[sam_b1 + 32:sam_b1 + 64] - tr_baseline_b3

        if (sam_b1 + 96) < start_sample or (sam_b1 + 64) > stop_sample:
            tr_baseline_b4 = np.median(tr_[sam_b1 + 64:sam_b1 + 96])
        elif (sam_b1 + 64) > start_sample and (sam_b1 + 96) < stop_sample:
            # fully contained... use the average block
            tr_baseline_b4 = tr_baseline
        elif (sam_b1 + 64) > start_sample and (sam_b1 + 96) > stop_sample:
            # tr_baseline_b4 = np.median(tr_[stop_sample:sam_b1+96])
            tr_baseline_b4 = tr_baseline
        elif (sam_b1 + 64) < start_sample and (sam_b1 + 96) > stop_sample:
            # tr_baseline_b4 = np.median([tr_[sam_b1+64:start_sample],tr_[stop_sample:sam_b1+96]])
            tr_baseline_b4 = tr_baseline
        elif (sam_b1 + 64) < start_sample and (sam_b1 + 96) < stop_sample:
            # tr_baseline_b4 = np.median(tr_[sam_b1+64:start_sample])
            tr_baseline_b4 = tr_baseline
        else:
            print("Shouldn't reach here")
            tr_baseline_b4 = tr_baseline
        tr_[sam_b1 + 64:sam_b1 + 96] = tr_[sam_b1 + 64:sam_b1 + 96] - tr_baseline_b4

        mean_c = np.mean(tr_[start_sample:stop_sample + 1])  # -tr_baseline)

        if verbose:
            print("Integration window {}, mean charge {:.1f} ADC (subtract baseline {:.1f} ADC)".format(int_samples,
                                                                                                        mean_c,
                                                                                                        tr_baseline))
        if show:
            # ax.plot(sams, tr_-tr_baseline)
            ax.plot(sams, tr_)
            ax.axvline(start_sample, ls="--", alpha=0.5)
            ax.axvline(stop_sample, ls="--", alpha=0.5)
            ax.set_xlabel("Sample")
            ax.set_ylabel("ADC [subtract basline]")
            ax.set_title(title)
            # if phases is not None:
            ax.axvline(sam_b1, ls="--", alpha=0.5, color='k')
            ax.axvline(sam_b1 + 32, ls="--", alpha=0.5, color='k')
            ax.axvline(sam_b1 + 64, ls="--", alpha=0.5, color='k')
            ax.axvline(sam_b1 + 98, ls="--", alpha=0.5, color='k')
            if ylim is not None:
                ax.set_ylim(ylim)
    return ampl[ievt, modInd, asic, ch, :], mean_c, sam_b1, start_sample, stop_sample


def read_raw_signal(reader, events=range(10), numBlock=4, nchannel=16,
                    nasic=4, chPerPacket=32, ADC_cut=None, get_timestamp=False,
                    modList=[1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124,
                             125, 126],
                    OUTDIR=OUTDIR,
                    ):
    nModules = len(modList)
    nEvents = len(events)
    print("{} events are going to be read".format(nEvents))
    ampl = np.zeros([nEvents, nModules, nasic, nchannel, nSamples])
    blocks = np.zeros(nEvents)
    phases = np.zeros(nEvents)
    if get_timestamp:
        timestamps = np.zeros(nEvents)
    data_ = event_reader(reader, events)
    icount = 0
    evt_dict={} #key is actual evt number, value is array index; this is needed because each event is revisted for diff mod asics etc
    prev_evt = 0
    ipix=0
    if events[0] != 0:
        print("Note that starting event is not 0")
    for ievt, modInd, asic, ch, sample, wf_, blockNumber, blockPhase, timestamp in data_:
            if icount == 0 and not evt_dict:
                prev_evt = ievt
                evt_dict[ievt] = 0
                blocks[0]=blockNumber
                phases[0]=blockPhase
            if prev_evt != ievt:
                #print("{} pixels found in event {} mod {} asic {} ch {}, filled {}".format(ipix, prev_evt, modList[modInd],asic, ch, icount))
                ipix=0
                #print(np.mean(ampl[icount, :, :, :, :]))
                if ievt not in evt_dict:
                    icount = np.max(list(evt_dict.values()))+1
                    evt_dict[ievt] = icount
                    blocks[icount]=blockNumber
                    phases[icount]=blockPhase
                else:
                    icount = evt_dict[ievt]
                prev_evt = ievt
                if icount>(nEvents-1):
                    print("shouldn't reach here")
                    break

            ipix+=1
            #print("Filling {}".format(icount))
            #print(icount, ievt, modInd, asic, ch, sample, blockNumber, blockPhase)
            ampl[icount, modInd, asic, ch, sample] = wf_
            if get_timestamp:
                timestamps[icount] = timestamp

        #else:
        #    ampl[ievt, modInd, asic, ch, sample] = wf_
    print("{} events read".format(nEvents))
    if get_timestamp:
        return timestamps, ampl, blocks, phases
    return ampl, blocks, phases


def read_raw_signal_evtloop_first(reader, events=range(10), numBlock=4, nchannel=16,
                    nasic=4, chPerPacket=32,
                    modList=[1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124,
                             125, 126],
                    OUTDIR=OUTDIR,
                    ):
    nModules = len(modList)
    nEvents = len(events)
    print("{} events are going to be read".format(nEvents))
    ampl = np.zeros([nEvents, nModules, nasic, nch, nSamples])
    blocks = np.zeros(nEvents)
    phases = np.zeros(nEvents)
    data_ = event_reader_evtloop_first(reader, events)
    icount = 0
    evt_dict={} #key is actual evt number, value is array index; this is needed because each event is revisted for diff mod asics etc
    prev_evt = 0
    ipix=0
    if events[0] != 0:
        print("Note that starting event is not 0")
    for ievt, modInd, asic, ch, sample, wf_, blockNumber, blockPhase, timestamp in data_:
            if icount == 0 and not evt_dict:
                prev_evt = ievt
                evt_dict[ievt] = 0
                blocks[0]=blockNumber
                phases[0]=blockPhase
            if prev_evt != ievt:
                #print("{} pixels found in event {} mod {} asic {} ch {}, filled {}".format(ipix, prev_evt, modList[modInd],asic, ch, icount))
                ipix=0
                #print(np.mean(ampl[icount, :, :, :, :]))
                if ievt not in evt_dict:
                    icount = np.max(evt_dict.values())+1
                    evt_dict[ievt] = icount
                    blocks[icount]=blockNumber
                    phases[icount]=blockPhase
                else:
                    icount = evt_dict[ievt]
                prev_evt = ievt
                if icount>(nEvents-1):
                    print("shouldn't reach here")
                    break

            ipix+=1
            #print("Filling {}".format(icount))
            #print(icount, ievt, modInd, asic, ch, sample, blockNumber, blockPhase)
            ampl[icount, modInd, asic, ch, sample] = wf_

        #else:
        #    ampl[ievt, modInd, asic, ch, sample] = wf_
    print("{} events read".format(nEvents))
    return ampl, blocks, phases

# diagnostic
def plot_traces(ampl_ped5k, ievt, mods=range(nModules), asics = range(nasic), channels=range(nchannel),
                blocks=None, phases=None,
                ylim=None,
                show=True, out_prefix="traces", interactive=False):
    # this is across 24 mod x 4 asic x 16 chan = 1536 channels
    # stability across all 512 blocks for each channel
    # pedestal cube should contain for each pixel and each block 1 value (assuming it's sample invariant)
    #ped_cube = np.zeros((nModules, nasic, nchannel, 512))
    #ped_var_cube = np.zeros((nModules, nasic, nchannel, 512))

    allmod_peaks = []
    for modInd in mods:
        nplot = 0
        thismod_peaks = []
        for asic in asics:
            if show:
                fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                nplot += 1
            for ch in channels:
                if show:
                    ax=axes.flatten()[ch]
                else:
                    ax=None
                if blocks is not None and phases is not None:
                    trace, mean_c, sam_b1, start_sample, stop_sample = get_trace_window_block_test(ampl_ped5k, ievt, modInd, asic, ch, blocks=blocks,
                                                        title="ch {}".format(ch), ax=ax, ylim=ylim,
                                                        phases=phases, show=show)
                    thismod_peaks.append(mean_c)
                else:
                    trace = get_trace(ampl_ped5k, ievt, modInd, asic, ch,
                                      show=show, ax=ax, title="ch {}".format(ch), ylim=ylim)
                    thismod_peaks.append(np.mean(trace))

            if show:
                plt.title("Mod {} Asic {}".format(modList[modInd], asic))
                plt.tight_layout()
                if interactive:
                    plt.show()
                else:
                    plt.savefig(OUTDIR + out_prefix+"_mod{}_asic{}.png".format(modList[modInd], asic, ch))
        allmod_peaks.append(thismod_peaks)
        if len(thismod_peaks)>0:
            plt.figure()
            plt.hist(thismod_peaks)
            plt.xlabel("Mean ADC")
            plt.title("Mod {}, {} channels".format(modList[modInd], len(thismod_peaks)))
            plt.savefig(OUTDIR + out_prefix + "_mod{}_meanADC_hist.png".format(modList[modInd]))


    #gs = gridspec.GridSpec(5,5)
    #gs.update(wspace=0.04, hspace=0.04)
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))

    for modInd in mods:
        loc, locReflect = calcLoc(modInd)
        thismod_peaks = allmod_peaks[modInd]
        #ax = plt.subplot(gs[loc])
        #ax = axes[loc]
        ax = axes[locReflect]
        ax.hist(thismod_peaks)
        ax.set_xlabel("Mean ADC")
        if ylim is not None:
            ax.set_xlim(ylim)
        # take off axes
        #ax.axis('off')
        #ax.set_aspect('equal')
        ax.set_title("Mod {}, {} channels".format(modList[modInd], len(thismod_peaks)))
    axes[2, 2].axis('off')
    plt.tight_layout()
    plt.savefig(OUTDIR + out_prefix + "_allmods_meanADC_hist.png")




def get_charge_distr_channel(ampl, modInd, asic, ch, sample,
                             blocks=None, choose_block=None,
                             show=False, out_prefix="pedestal",
                             ax=None, xlim=None):
    print("Looking at mod {}, asic {}, ch {}, sample {}".format(modList[modInd], asic, ch, sample))
    if blocks is not None and choose_block is not None:
        #print("Chose block {}".format(choose_block))
        indices = np.where(blocks == choose_block)
        #print("Indices")
        #print(indices)
        cs = ampl[indices, modInd, asic, ch, sample].flatten()
        #print("charges")
        #print(cs)
    elif sample == -1:
        #average over samples
        cs = np.median(ampl[:, modInd, asic, ch, :], axis=1).flatten()
    else:
        cs = ampl[:, modInd, asic, ch, sample].flatten()
    if show:
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(cs, bins="auto", density=False)
        ax.set_xlabel("ADC")
        ax.set_ylabel("# of evts")
        if blocks is not None and choose_block is not None:
            ax.set_title("mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                        ch) +"_sample"+str(sample) + "_block" + str(choose_block))
        else:
            ax.set_title("mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                ch) +"_sample"+str(sample))
        if xlim is not None:
            ax.set_xlim(xlim)
        plt.tight_layout()
        if out_prefix is not None:
            if blocks is not None and choose_block is not None:
                plt.savefig(
                    OUTDIR + "/" + out_prefix + "_mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                        ch) +"_sample"+str(sample) + "_block" +str(choose_block)+ ".png")
            else:
                plt.savefig(OUTDIR+"/"+out_prefix+"_mod"+str(modList[modInd])+"_asic"+str(asic)+"_ch"+str(ch)+"_sample"+str(sample)+".png")
    return cs


def plot_charge_distr_channel(ampl, modInd, asic, ch, samples='all',
                              blocks=None, choose_block=None,
                             show=False, median=True, out_prefix="charge_distr"):
    if samples == 'all':
        samples = list(range(ampl.shape[-1]))
    mean_cs = np.zeros(len(samples))
    std_cs = np.zeros(len(samples))
    for i, sample in enumerate(samples):
        if blocks is not None and choose_block is not None:
            indices = np.where(blocks==choose_block)
            cs = ampl[indices, modInd, asic, ch, sample]
        else:
            cs = ampl[:, modInd, asic, ch, sample]
        if median:
            mean_cs[i] = np.median(cs)
        else:
            mean_cs[i] = np.mean(cs)
        std_cs[i] = np.std(cs)
    if show:
        plt.errorbar(samples, mean_cs, std_cs, fmt='.')
        plt.xlabel("Sample")
        plt.ylabel("ADC")
        if blocks is not None and choose_block is not None:
            plt.title("mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                        ch) + "_block" + str(choose_block))
        else:
            plt.title("mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                ch) )
        plt.tight_layout()
        if out_prefix is not None:
            if blocks is not None and choose_block is not None:
                plt.savefig(
                    OUTDIR + "/" + out_prefix + "_mod" + str(modList[modInd]) + "_asic" + str(asic) + "_ch" + str(
                        ch) + "_block" +str(choose_block)+ ".png")
            else:
                plt.savefig(OUTDIR+"/"+out_prefix+"_mod"+str(modList[modInd])+"_asic"+str(asic)+"_ch"+str(ch)+".png")

    return mean_cs, std_cs


def ped_block_distr_vectorized(ampl_ped5k, blocks5k):
    # this is across 24 mod x 4 asic x 16 chan = 1536 channels
    # stability across all 512 blocks for each channel
    # pedestal cube should contain for each pixel and each block 1 value (assuming it's sample invariant)
    ped_cube = np.zeros((nModules, nasic, nchannel, 512))
    ped_var_cube = np.zeros((nModules, nasic, nchannel, 512))

    for block_ in range(512):
        cs = np.median(ampl_ped5k[blocks5k == block_, :, :, :, :], axis=4)
        cs = np.median(cs, axis=0)
        ped_cube[:, :, :, block_] = np.median(cs)
        ped_var_cube[:, :, :, block_] = np.std(cs)

    return ped_cube, ped_var_cube


def ped_block_sample_vectorized(ampl_ped5k, blocks5k):
    # this is across 24 mod x 4 asic x 16 chan = 1536 channels
    # stability across all 512 blocks for each channel
    # pedestal cube should contain for each pixel and each block 1 value (assuming it's sample invariant)
    ped_cube = np.zeros((nModules, nasic, nchannel, nSamples, 512))
    ped_var_cube = np.zeros((nModules, nasic, nchannel, nSamples, 512))

    for block_ in range(512):
        cs = np.median(ampl_ped5k[blocks5k == block_, :, :, :, :], axis=0)
        stds = np.std(ampl_ped5k[blocks5k == block_, :, :, :, :], axis=0)
        ped_cube[:, :, :, :, block_] = cs #np.median(cs)
        ped_var_cube[:, :, :, :, block_] = stds

    return ped_cube , ped_var_cube


#def pedestal_subtraction(ampl, blocks, ped_cube):


# UI
def show_image(red_ampl, show=True, minZ=1, maxZ=1000, simple_baseline=True, outfile=None):
    if simple_baseline:
        baseline = np.mean(red_ampl[:, :, :, 1:15], axis=3)
    else:
        baseline = np.zeros(red_ampl[:, :, :, 0].shape)
    peak = np.amax(red_ampl[:, :, :, 20:], axis=3)
    diff = peak - baseline
    allsamp_diff = red_ampl - np.tile(np.expand_dims(baseline, axis=3), nSamples)
    for modInd in range(len(diff)):
        if modInd in range(9):
            diff[modInd, :, :] /= 2.0
            allsamp_diff[modInd, :, :, :] /= 2.0

    heatArray = diff.reshape((len(modList), 64))
    #redAmplArray = allsamp_diff.reshape((nModules, nasic * nchannel, nSamples))
    ImArr = np.zeros((40, 40))
    physHeatArr = np.zeros([nModules, 8, 8])
    physHeatArr[:, row, col] = heatArray

    for modInd in range(nModules):
        loc, locReflect = calcLoc(modInd)
        if loc[1] % 2 == 0:
            physHeatArr[modInd, :, :] = np.rot90(physHeatArr[modInd, :, :], k=2)
        ImArr[(5 - posGrid[modPos[modList[modInd]]][0]) * 8:(6 - posGrid[modPos[modList[modInd]]][0]) * 8,
        (5 - posGrid[modPos[modList[modInd]]][1]) * 8:(6 - posGrid[modPos[modList[modInd]]][1]) * 8] = np.fliplr(
            physHeatArr[modInd, :, :])
    if show:
        plt.figure()
        ax = plt.subplot(111)
        cx = plt.pcolor(ImArr, vmin=minZ, vmax=maxZ)
        plt.colorbar()
        ax.set_aspect('equal')
    if outfile is not None:
        plt.savefig(outfile)
    return ImArr



# analysis


def gaussian(height, center_x, center_y, width_x, width_y, rotation, baseline=0):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)

    # center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    # center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        # xp = x * np.cos(rotation) - y * np.sin(rotation)
        # yp = x * np.sin(rotation) + y * np.cos(rotation)
        xp = (x - center_x) * np.cos(rotation) - (y - center_y) * np.sin(rotation)  # + center_x
        yp = (x - center_x) * np.sin(rotation) + (y - center_y) * np.cos(rotation)  # + center_y
        g = height * np.exp(
            -(((-xp) / width_x) ** 2 +
              ((-yp) / width_y) ** 2) / 2.) + baseline
        # -(((center_x-xp)/width_x)**2+
        #  ((center_y-yp)/width_y)**2)/2.)
        return g

    return rotgauss


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    central moments """
    # M00
    total = data.sum()
    X, Y = np.indices(data.shape)
    # centroid
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    # M2s
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y, theta)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = sp.optimize.leastsq(errorfunction, params)
    return p, success


def fit_gaussian2d(data, outfile=None):  # , amp=1, xc=0,yc=0,A=1,B=1,theta=0, offset=0):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # plt.matshow(data, cmap=plt.cm.gray)
    # plt.pcolor(data, cmap=plt.cm.gray)
    cax = plt.pcolor(data, cmap=plt.cm.gray)
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    params, success = fitgaussian(data)
    fit = gaussian(*params)

    # plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper, levels=[68, 90, 95])
    ax = plt.gca()
    (height, x, y, width_x, width_y, theta) = params
    # print(height, x, y, width_x, width_y, np.rad2deg(theta))
    width_x = np.abs(width_x)
    width_y = np.abs(width_y)
    if width_x > width_y:
        width_x, width_y = width_y, width_x
        theta = theta + 90

    # plt.plot([x,x], [y,y+width_y], ls='-', color='c')
    # plt.plot([x,x+width_x], [y,y], ls='-', color='c')
    e = Ellipse(xy=np.array([y, x]), width=width_y * 2,
                height=width_x * 2, angle=theta, linewidth=1, fill=False, alpha=0.9)
    ax.add_artist(e)
    e.set_color('c')
    e2 = Ellipse(xy=np.array([y, x]), width=width_y * 4,
                 height=width_x * 4, angle=theta, linewidth=1, fill=False, alpha=0.9)
    ax.add_artist(e2)
    e2.set_color('c')

    plt.plot([20, y], [20, x], ls='--', alpha=0.6)

    dist = np.sqrt((20 - y) ** 2 + (20 - x) ** 2)
    # if theta<-180 or theta> 360:
    # print(theta)
    theta = (theta + 180) % 360 - 180
    # print(theta)

    plt.plot([y - 10 * np.cos(np.deg2rad(theta)), y + 10 * np.cos(np.deg2rad(theta))],
             [x - 10 * np.sin(np.deg2rad(theta)), x + 10 * np.sin(np.deg2rad(theta))], ls='--', alpha=0.6)

    # print(theta)
    # print(np.rad2deg(np.arctan2((y-20.), (x-20))))

    center_yx = np.array([20, 20])
    centroid_yx = np.array([y, x])
    ma_yx = np.array([y + 10 * np.cos(np.deg2rad(theta)), x + 10 * np.sin(np.deg2rad(theta))])
    # ma_yx = np.array([y - 10 * np.cos(np.deg2rad(theta)),x - 10 * np.sin(np.deg2rad(theta))])

    plt.plot([center_yx[0], centroid_yx[0], ma_yx[0]],
             [center_yx[1], centroid_yx[1], ma_yx[1]], 'ro', ls='--', alpha=0.6)

    line_c = center_yx - centroid_yx
    line_ma = ma_yx - centroid_yx

    cosine_alpha = np.dot(line_c, line_ma) / (np.linalg.norm(line_c) * np.linalg.norm(line_ma))
    alpha = np.degrees(np.arccos(cosine_alpha))
    if alpha > 90 and alpha < 180:
        alpha = 180 - alpha
    plt.text(0.95, 0.05, """
            alpha : %.1f
            $\sigma_x$ : %.1f
            $\sigma_y$ : %.1f""" % (alpha, width_x, width_y),
             fontsize=14, horizontalalignment='right', alpha=0.8,
             verticalalignment='bottom', transform=ax.transAxes, color='y')

    print(
        "peak {}, cenX {:.2f}, cenY {:.2f}, width {:.2f}, length {:.2f}, theta {:.2f}, dist {:.2f}, alpha {:.2f}".format(
            height, x, y, width_x, width_y, theta, dist, alpha))
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)

    return height, x, y, width_x, width_y, theta, dist, alpha, success


# application
def ped_block_distr(ampl_ped5k, blocks5k, sample=-1, show=False, out_prefix="pedestal_block_distr"):
    # this is across 24 mod x 4 asic x 16 chan = 1536 channels
    # stability across all 512 blocks for each channel
    # pedestal cube should contain for each pixel and each block 1 value (assuming it's sample invariant)
    ped_cube = np.zeros((nModules, nasic, nchannel, 512))
    ped_var_cube = np.zeros((nModules, nasic, nchannel, 512))
    for modInd in range(nModules):
        for asic in range(nasic):
            for ch in range(nchannel):
                #ped_mean = np.zeros(512)
                #ped_median = np.zeros(512)
                #ped_std = np.zeros(512)
                cblock_ = 0

                for subp_ in range(512 / 16):
                    if show:
                        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                    #for i, ax in enumerate(axes.flatten()):
                    for i in range(16):
                        if show:
                            ax = axes.flatten()[i]
                            cs_ped = get_charge_distr_channel(ampl_ped5k, modInd, asic, ch, sample, show=True, out_prefix=None,  # "pedestal328534",
                                                          blocks=blocks5k, choose_block=cblock_, ax=ax, xlim=[300, 700])
                        else:
                            cs_ped = get_charge_distr_channel(ampl_ped5k, modInd, asic, ch, sample, show=False, out_prefix=None,
                                                              # "pedestal328534",
                                                              blocks=blocks5k, choose_block=cblock_, )
                        #ped_mean[cblock_] = np.mean(cs_ped)
                        ped_cube[modInd, asic, ch, cblock_] = np.median(cs_ped)
                        ped_var_cube[modInd, asic, ch, cblock_]  = np.std(cs_ped)
                        cblock_ += 1
                    if show:
                        plt.tight_layout()
                        plt.savefig(OUTDIR + out_prefix+"_mod{}_asic{}_ch{}_allblocks_sample_median_page{}.png".format(modList[modInd], asic, ch, subp_ + 1))
                    # break
                if show:
                    fig, ax = plt.subplots(1, 1)

                    ax.errorbar(range(512), ped_cube[ modInd, asic, ch, :], ped_var_cube[ modInd, asic, ch, :], fmt='.')
                    plt.xlabel("Block")
                    plt.ylabel("ADC median")
                    plt.tight_layout()
                    plt.savefig(OUTDIR + out_prefix+"_mod{}_asic{}_ch{}_allblocks_sample_median_median.png".format(modList[modInd], asic, ch))

    return ped_cube, ped_var_cube

# some diagnositcs with no re-usability:
def block_stability(ampl_ped5k, blocks5k):
    # stability across blocks
    ped_mean = np.zeros(512)
    ped_median = np.zeros(512)
    ped_std = np.zeros(512)
    cblock_ = 0

    for subp_ in range(512 / 16):

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        for i, ax in enumerate(axes.flatten()):
            cs_ped = get_charge_distr_channel(ampl_ped5k, 1, 1, 5, 0, show=True, out_prefix=None,  # "pedestal328534",
                                              blocks=blocks5k, choose_block=cblock_, ax=ax, xlim=[300, 700])
            ped_mean[cblock_] = np.mean(cs_ped)
            ped_median[cblock_] = np.median(cs_ped)
            ped_std[cblock_] = np.std(cs_ped)

            cblock_ += 1
        plt.tight_layout()
        plt.savefig(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_sample0_allblocks_page{}.png".format(subp_ + 1))
        # break

    fig, ax = plt.subplots(1, 1)

    ax.errorbar(range(512), ped_mean, ped_std, fmt='.')
    plt.xlabel("Block")
    plt.ylabel("ADC mean")
    plt.tight_layout()
    plt.savefig(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_block_stability_allsamples_mean.pdf")

    fig, ax = plt.subplots(1, 1)

    ax.errorbar(range(512), ped_median, ped_std, fmt='.')
    plt.xlabel("Block")
    plt.ylabel("ADC median")
    plt.tight_layout()
    plt.savefig(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_block_stability_allsamples_median.pdf")


def sample_stability(ampl_ped5k):
    # stability across samples; slow
    pedsam_mean = np.zeros(nSamples)
    pedsam_median = np.zeros(nSamples)
    pedsam_std = np.zeros(nSamples)
    sam = 0
    for subp_ in range(128 / 16):

        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        for i, ax in enumerate(axes.flatten()):
            cs_ped = get_charge_distr_channel(ampl_ped5k, 1, 1, 5, sam, show=True, out_prefix=None,
                                              # "pedestal328534",
                                              ax=ax, xlim=[300, 700])
            pedsam_mean[sam] = np.mean(cs_ped)
            pedsam_median[sam] = np.median(cs_ped)
            pedsam_std[sam] = np.std(cs_ped)
            sam += 1
        plt.tight_layout()
        plt.savefig(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_allblocks_each_sample_page{}.png".format(subp_ + 1))
        # break

    fig, ax = plt.subplots(1, 1)

    ax.errorbar(range(nSamples), pedsam_mean, pedsam_std, fmt='.')
    plt.xlabel("Sample")
    plt.ylabel("ADC mean")
    plt.tight_layout()
    plt.save(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_allblocks_allsamples_mean.pdf")

    fig, ax = plt.subplots(1, 1)

    ax.errorbar(range(nSamples), pedsam_median, pedsam_std, fmt='.')
    plt.xlabel("Sample")
    plt.ylabel("ADC median")
    plt.tight_layout()
    plt.save(OUTDIR + "pedestal328534_5k_mod2_asic1_ch5_allblocks_allsamples_median.pdf")


def find_bright_events(ampl_crab1k):
    from scipy.signal import medfilt2d
    evts = []
    pulseheights = []
    xs = []
    ys = []
    widths = []
    lengths = []
    thetas = []
    dists = []
    alphas = []
    for i in range(evt_, evt_ + 1000):
        #test_slice = ampl_crab1k[i, :, :, :, 40:80]
        # if np.percentile(test_slice[(test_slice<3800) & (test_slice>500)], 99) < 1500:
        #    continue
        # print(i, np.percentile(test_slice[(test_slice<3800) & (test_slice>500)], 99),
        #      np.median(test_slice[(test_slice<3800) & (test_slice>500)]),
        #     )#np.max(test_slice[(test_slice<3800) & (test_slice>500)]))

        im7 = show_image(ampl_crab1k[i], maxZ=4000, show=False)
        im7_smooth = medfilt2d(im7, 3)
        if np.max(im7_smooth) < 1400:
            continue
        print(i, np.percentile(im7_smooth, 99),
              np.median(im7_smooth), np.max(im7_smooth))
        plt.figure()
        ax = plt.subplot(111)
        cx = plt.pcolor(im7_smooth, vmin=1, vmax=4000)
        pulseheight, x, y, width, length, theta, dist, alpha, success = fit_gaussian2d(im7_smooth)
        evts.append(i)
        pulseheights.append(pulseheight)
        xs.append(x)
        ys.append(y)
        widths.append(width)
        lengths.append(length)
        thetas.append(theta)
        dists.append(dist)
        alphas.append(alpha)
        plt.colorbar()
        ax.set_aspect('equal')
        show_image(ampl_crab1k[i], maxZ=4000, show=True,
                   outfile=OUTDIR + "image_run328540_evt{}.pdf".format(i))
    return evts, pulseheights, xs, ys, widths, lengths, thetas, dists, alphas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('evt', type=int, default=7, help="Start event number")
    parser.add_argument('-n', '--num_evt', type=int, default=1, help="Number of events to read.Default is 1.")
    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run
    evt_num = args.evt
    n_evts = int(args.num_evt)
    print("Reding {} events starting from evt {} in run {}".format(n_evts, evt_num, run_num))

    reader = get_reader(run_num)
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    ampl, blocks, phases = read_raw_signal(reader, range(evt_num, evt_num+n_evts))
    np.save("run{}_evt{}to{}_amplitude.npy".format(run_num, evt_num, evt_num+n_evts), ampl)
    np.save("run{}_evt{}to{}_blocks.npy".format(run_num, evt_num, evt_num+n_evts), blocks)
    np.save("run{}_evt{}to{}_phases.npy".format(run_num, evt_num, evt_num+n_evts), phases)
    #sys.exit(1)

    im = show_image(ampl[0])
    _ = fit_gaussian2d(im)
    plt.savefig("test_image.png")

    if np.median(im)<10:
        print("this event is probably no good, exit")
        exit()
    plot_traces(ampl, 0, mods=range(nModules), asics = range(nasic), channels=range(nchannel),
                show=True, out_prefix="traces_{}_evt{}".format(run_num, evt_num))

    """
    #read 5k pedestal events: 
    run_num_pedestal = 328534
    reader_pedestal = get_reader(run_num_pedestal)
    ampl_ped5k, blocks5k, phases5k = read_raw_signal(reader_pedestal, range(5000))
    np.save("run328534_pedestal_amplitude.npy", ampl_ped5k)
    np.save("run328534_pedestal_blocks.npy", blocks5k)
    np.save("run328534_pedestal_phases.npy", phases5k)
    
    #read 1k crab events: 
    reader_crab = get_reader(run_num)
    ampl_crab1k, blocks_crab1k, phases_crab1k = read_raw_signal(reader_crab, range(1000))
    np.save("run{}_crab_amplitude1k.npy".format(run_num), ampl_crab1k)
    np.save("run{}_crab_blocks1k.npy".format(run_num), blocks_crab1k)
    np.save("run{}_crab_phases1k.npy".format(run_num), phases_crab1k)

    
    #raw charge distr
    # ped
    _ = plt.hist(ampl_ped5k.flatten(), bins=np.arange(0,2500,10))
    plt.xlabel("ADC")
    plt.savefig(OUTDIR+"ped_raw_charge_distr_5k_run328534.pdf")
    # crab run
    _ = plt.hist(ampl_crab1k.flatten(), bins=np.arange(0,3500,10))
    plt.xlabel("ADC")
    plt.savefig(OUTDIR+"raw_charge_distr_crab_1k_run328540.pdf")

    #get pedestal maps for all blocks:
    ped_cube5k, ped_var_cube5k = ped_block_distr_vectorized(ampl_ped5k, blocks5k)
    np.save("run328534_pedestal_block_cube.npy", ped_cube5k)
    np.save("run328534_pedestal_std_block_cube.npy", ped_var_cube5k)

    #plot all traces of each pixel:
    #crab run evt 7
    plot_traces(ampl_crab10, 7, mods=range(nModules), asics = range(nasic), channels=range(nchannel),
                show=True, out_prefix="traces_{}_evt{}".format(run_num, 7))
                
    #pedestal; note this should be using blocks that correspond to data, not the simple block 1 as shown below
    ped_sub_evt7 = ampl_crab10[7] - np.tile(np.expand_dims(ped_cube5k[:, :, :, 1], axis=3), nSamples)
    plot_traces(np.expand_dims(ped_sub_evt7, axis=0), 0,
                mods=range(nModules), asics=range(nasic), channels=range(nchannel),
                show=True, out_prefix="ped_sub1_traces_{}_evt{}".format(run_num, 7))
    """