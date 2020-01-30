import binascii
import pandas as pd

DATADIR='/data/local_outputDir/'

def get_mod_trigger_pattern_array(h):
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
        ax = axes[np.where(FEE_map==thismod)[0][0], np.where(FEE_map==thismod)[1][0]]
        ax.imshow(hitarr_)
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
        ax = axes[np.where(FEE_map == thismod)[0][0], np.where(FEE_map == thismod)[1][0]]
        cx = ax.imshow(hitarr_dict[thismod], vmin=0, vmax=50)
        plt.colorbar(cx, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Mod {}".format(thismod))
    # break

    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot trigger hitmap')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('-t', '--thresh', type=int, default=120, help="Thresh to plot")
    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run

    df = pd.read_csv(DATADIR+"_hitmaps.txt".format(run_num), sep=r"\s+", header=None)
    plot_50trigger_hitmaps(df[df[0] == args.thresh])
    plt.show()