import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml

#OUTDIR='./'
OUTDIR='/data/analysis_output/trigger_hitmaps/'


def plot_masked_trigger_pixels(masked_trigger_pixels, cmap=plt.cm.RdBu, outfile=None):
    FEE_map = np.array([[4, 5, 1, 3, 2],
                        [103, 125, 126, 106, 9],
                        [119, 108, 110, 121, 8],
                        [115, 123, 124, 112, 7],
                        [100, 111, 114, 107, 6]
                        ])
    #list_pi7 = FEE_map.flatten()[::-1]

    mapping_array = np.array([[3, 0], [3, 1], [2, 0], [2, 1],
                              [3, 2], [3, 3], [2, 2], [2, 3],
                              [1, 0], [1, 1], [0, 0], [0, 1],
                              [1, 2], [1, 3], [0, 2], [0, 3],
                              ])

    if len(masked_trigger_pixels) == 0:
        print("No masked pixels")
        return
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))

    i = 0
    for thismod in FEE_map.flatten():
        #print(thismod)
        if thismod == 110:
            continue
        ax = axes[np.where(FEE_map == thismod)[0][0], 4 - np.where(FEE_map == thismod)[1][0]]
        if thismod in masked_trigger_pixels.keys():
            subarr = np.zeros((4, 4))
            for pix in masked_trigger_pixels[thismod]:
                subarr[mapping_array[pix, 0], mapping_array[pix, 1]] = -1

            ax.imshow(subarr, cmap=cmap)
        else:
            ax.imshow(np.zeros((4, 4)), cmap=cmap, vmax=0, vmin=-1)
        ax.set_title("Mod {}".format(thismod))
        ax.axis('off')
        i = i + 1
        # break
    axes[2, 2].axis('off')
    ax = axes[2, 2]
    for index_, xy in enumerate(mapping_array):
        text_y = 3 - xy[0] + 0.5
        text_x = xy[1] + 0.5
        ax.text(text_x, text_y, index_, color='black', ha='center', va='center', fontsize=18)
    ax.set_title("Trigger Pixel Mapping")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot trigger hitmap')
    parser.add_argument('-f', '--filename', default="masked_trigger_pixels.yml", help="Mask yaml file to plot. Default to masked_trigger_pixels.yml")
    parser.add_argument('--outdir', default=None, help="Default to  dir {}".format(OUTDIR))
    parser.add_argument('-o', '--outfile', default=None, help="Figure file to save plot to. ")
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")

    args = parser.parse_args()


    if args.outdir is not None:
        OUTDIR = args.outdir
    show = args.interactive

    with open(args.filename, 'r') as pixels_file:
        masked_trigger_pixels = yaml.safe_load(pixels_file)
    if masked_trigger_pixels is None:
        masked_trigger_pixels = {}

    plot_masked_trigger_pixels(masked_trigger_pixels, outfile=args.outfile)

    if show:
        plt.show()