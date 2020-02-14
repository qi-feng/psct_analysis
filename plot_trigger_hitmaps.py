import binascii
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time




DATADIR='/data/local_outputDir/'
#DATADIR='../diagnositcs/trigger_hitmaps/'
#OUTDIR='./'
OUTDIR='/data/analysis_output/trigger_hitmaps/'

def get_mod_trigger_pattern_array(h):
    h_size = 16
    h = ( bin(int(h, 16))[2:] ).zfill(h_size)
    subarr = np.zeros((4,4))
    mapping_array = np.array([ [3, 0], [3, 1], [2, 0], [2, 1],
                              [3, 2], [3, 3], [2, 2], [2, 3],
                              [1, 0], [1, 1], [0, 0], [0, 1],
                              [1, 2], [1, 3], [0, 2], [0, 3],
                            ])

    for i_reverse, trigger_ in enumerate(h):
        i = 15-i_reverse
        #print(i, trigger_)
        #print(mapping_array[i,0], mapping_array[i,1])
        subarr[mapping_array[i,0], mapping_array[i,1]] = float(trigger_)
        #subarr = subarr.T
    return subarr

def get_mod_trigger_pattern_array_old(h):
    h = ( bin(int(h, 16))[2:] ).zfill(h_size)
    subarr = np.zeros((4,4))

    for i, trigger_ in enumerate(h):
        #print(i, trigger_)
        subarr[3-i//4, i%4] = float(trigger_)
        subarr = subarr.T
    return subarr


def plot_trigger_hitmap(th):
    FEE_map = np.array([[4,5,1,3,2],
               [103,125,126,106,9],
               [119, 108, 110, 121, 8],
               [115, 123, 124, 112, 7],
               [100,111,114,107,6]
              ])

    list_pi7=FEE_map.flatten()[::-1]
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))

    i=0
    for th_ in th:
        hitarr_ = get_mod_trigger_pattern_array(str(th_))
        thismod = list_pi7[i]
        ax = axes[np.where(FEE_map==thismod)[0][0], 4-np.where(FEE_map==thismod)[1][0]]
        ax.imshow(hitarr_, interpolation='none')
        ax.set_title("Mod {}".format(thismod))
        i=i+1
        #break

    plt.tight_layout()


def plot_50trigger_hitmaps(ths):
    FEE_map = np.array([[4, 5, 1, 3, 2],
                        [103, 125, 126, 106, 9],
                        [119, 108, 110, 121, 8],
                        [115, 123, 124, 112, 7],
                        [100, 111, 114, 107, 6]
                        ])

    list_pi7 = FEE_map.flatten()[::-1]
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))
    hitarr_dict = {}

    for _, th in ths.iterrows():
        i = 0
        # print(th[1:26])
        for th_ in th[1:26]:
            hitarr_ = get_mod_trigger_pattern_array(str(th_))
            thismod = list_pi7[i]
            # print(thismod, hitarr_)
            if thismod not in hitarr_dict:
                hitarr_dict[thismod] = hitarr_
            else:
                hitarr_dict[thismod] += hitarr_
            i += 1

    for i in range(25):
        thismod = list_pi7[i]
        if thismod == 110:
            continue
        #ax = axes[np.where(FEE_map == thismod)[0][0], np.where(FEE_map == thismod)[1][0]]
        # reflect
        ax = axes[np.where(FEE_map==thismod)[0][0], 4-np.where(FEE_map==thismod)[1][0]]

        cx = ax.imshow(hitarr_dict[thismod], vmin=0, vmax=50, interpolation='none', cmap=plt.cm.viridis)#, origin="lower")
        plt.colorbar(cx, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Mod {}".format(thismod))
        ax.axis('off')
    # break
    axes[2, 2].axis('off')

    plt.tight_layout()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot trigger hitmap')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('-t', '--thresh', type=int, default=120, help="Thresh to plot. Default is 120. ")
    parser.add_argument('-a', '--all_thresh', action="store_true", help="Plot all threshold values, can take a while. ")
    #parser.add_argument('-s', '--save', action="store_true", help="Flag to save plots.")
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    parser.add_argument('--outdir', default=None, help="Default to  dir {}".format(OUTDIR))
    parser.add_argument('--datadir', default=None, help="Default to dir {}".format(DATADIR))

    start_time = time.time()
    args = parser.parse_args()

    if args.datadir is not None:
        DATADIR = args.datadir
    if args.outdir is not None:
        OUTDIR = args.outdir
    show = args.interactive

    #example just to read 10 evts and plot one
    run_num = args.run

    df = pd.read_csv(DATADIR+"{}_hitmaps.txt".format(run_num), sep=r"\s+", header=None)
    if args.all_thresh:
        for thresh in np.unique(df[0].values):
            print("Plotting the sum of 50 trigger hit maps for thresh {}".format(thresh))
            plot_50trigger_hitmaps(df[df[0] == thresh])
            plt.savefig(OUTDIR+"trigger_hitmaps_50sum_run{}_thresh{}.png".format(run_num, thresh))
    else:
        print("Plotting the sum of 50 trigger hit maps for thresh {}".format(args.thresh))
        plot_50trigger_hitmaps(df[df[0] == args.thresh])
        #if args.save:
        plt.savefig(OUTDIR+"trigger_hitmaps_50sum_run{}_thresh{}.png".format(run_num, args.thresh))
    if show:
        plt.show()
    elapsed_time = time.time() - start_time
    print("Done. Elapsed time: {} s".format(elapsed_time))

