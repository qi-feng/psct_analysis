from pSCT_analysis import *
from scipy.signal import medfilt2d
import time

DATADIR='/mnt/data476G/pSCT_data/'
OUTDIR='./'
norm_map_default = np.load("norm_map_default.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('--start_evt', type=int, default=0, help="Start event number")
    parser.add_argument('-n', '--num_evt', type=int, default=-1, help="Number of events to read. Default is all events.")
    parser.add_argument('--peak_ADC_lower', type=int, default=1400, help="Cut on peak ADC lower; anything with peak below this value is thrown away")
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    parser.add_argument('-s', '--save', action="store_true", help="Flag to save plots.")
    parser.add_argument('-f', '--flasher', action="store_true", help="Flag to search for flasher events.")
    parser.add_argument('--flatfield', action="store_true", help="Flag to try flatfielding.")
    parser.add_argument('-l', '--flasher_file', default="", help="File to save flasher event numbers.")
    parser.add_argument('--smooth', action="store_true", help="Show/save the smoothed image (using a 3x3 median kernel).")
    parser.add_argument('--outfile', default=None, help="Text file to save parameters to. ")
    parser.add_argument('--outdir', default=None, help="Default to current dir ")
    parser.add_argument('--datadir', default=None, help="Default to dir {}".format(DATADIR))

    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run
    evt_start = args.start_evt
    n_evts = int(args.num_evt)
    show = args.interactive

    if args.datadir is not None:
        DATADIR = args.datadir
    if not args.interactive and not args.save:
        print("You didn't specify either interactive or save. Assume interactive. ")
        show=True
    reader = get_reader(run_num, DATADIR=DATADIR)
    if n_evts == -1:
        n_evts = reader.GetNEvents()

    print("Reding {} events starting from evt {} in run {}".format(n_evts, n_evts, run_num))
    start_time = time.time()

    read_per_cycle = 100
    ncycles = n_evts//read_per_cycle + 1
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    evts = []
    pulseheights = []
    xs = []
    ys = []
    widths = []
    lengths = []
    thetas = []
    dists = []
    alphas = []
    if args.outdir is not None:
        OUTDIR = args.outdir
    if args.outfile is not None:
        colnames = ['evt_num', 'timestamp', 'pulse_height', 'centroid_x', 'centroid_y', 'width', 'length', 'dist', 'alpha']
        ofile = OUTDIR + "/" + args.outfile
        with open(ofile, 'w') as paramfileio:
            paramfileio.write(" ".join(colnames))
            paramfileio.write("\n")
    if args.flasher:
        if args.flasher_file=="":
            flasher_file = OUTDIR +"/flasher_evt_nums_run{}.npy".format(run_num)
        else:
            flasher_file = OUTDIR + args.flasher_file
        with open(flasher_file, 'w') as ffio:
            ffio.write("evt\n")
    current_evt = evt_start
    for icycle in range(ncycles):
        if icycle == (ncycles - 1):
            stop_evt = current_evt+n_evts
        else:
            stop_evt = current_evt + read_per_cycle
        print("Reading evt {} to {}...".format(current_evt, stop_evt))
        timestamps, ampl, blocks, phases = read_raw_signal(reader, range(current_evt, stop_evt), get_timestamp=True)

        for i in range(current_evt, stop_evt):
            im = show_image(ampl[i-current_evt], maxZ=4000, show=False)
            im_smooth = medfilt2d(im, 3)
            if np.percentile(im_smooth[im_smooth != 0], 90) > 500:
                print("This is probably a flasher event")
                isf = 'f'
                if args.flasher:
                    with open(flasher_file, 'a') as ffio:
                        ffio.write("{}\n".format(i))
            else:
                isf = ''
            if args.flasher and isf == '':
                continue
            elif not args.flasher and isf == 'f':
                # let's skip flashers
                continue
            elif np.max(im_smooth) < args.peak_ADC_lower and not args.flasher:
                continue
            #plt.figure()
            #ax = plt.subplot(111)
            #cx = plt.pcolor(im_smooth, vmin=1, vmax=4000)

            if args.flatfield:
                im = im / norm_map_default
                im_smooth = medfilt2d(im, 3)

            if args.save:
                if args.smooth:
                    if args.flasher:
                        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                        plt.pcolor(im_smooth, cmap=plt.cm.gray)
                        plt.xlim(0, 40)
                        plt.ylim(0, 40)
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(OUTDIR +"/smooth_image_run{}_evt{}.png".format(run_num, i))
                    else:
                        pulseheight, x, y, width, length, theta, dist, alpha = fit_gaussian2d(im_smooth, outfile=OUTDIR +"/smooth_image_fit_run{}_evt{}.png".format(run_num, i))
                else:
                    if args.flasher:
                        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                        plt.pcolor(im, cmap=plt.cm.gray)
                        plt.xlim(0, 40)
                        plt.ylim(0, 40)
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(OUTDIR +"/image_run{}_evt{}.png".format(run_num, i))
                    else:
                        pulseheight, x, y, width, length, theta, dist, alpha = fit_gaussian2d(im, outfile=OUTDIR +"/image_fit_run{}_evt{}.png".format(run_num, i))
                if show:
                    plt.colorbar()
                    plt.show()

                np.save(OUTDIR +"/ampl_run{}_evt{}_block{}_phase{}.npy".format(run_num, i, blocks[i - current_evt], phases[i - current_evt]), ampl[i - current_evt])
                np.save(OUTDIR +"/im_run{}_evt{}_block{}_phase{}.npy".format(run_num, i, blocks[i - current_evt], phases[i - current_evt]), im)
            else:
                if args.smooth:
                    if args.flasher:
                        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                        plt.pcolor(im_smooth, cmap=plt.cm.gray)
                        plt.xlim(0, 40)
                        plt.ylim(0, 40)
                        plt.tight_layout()
                    else:
                        pulseheight, x, y, width, length, theta, dist, alpha = fit_gaussian2d(im_smooth)
                else:
                    if args.flasher:
                        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                        plt.pcolor(im, cmap=plt.cm.gray)
                        plt.xlim(0, 40)
                        plt.ylim(0, 40)
                        plt.tight_layout()
                    else:
                        pulseheight, x, y, width, length, theta, dist, alpha = fit_gaussian2d(im)

                plt.show()
            #show_image(ampl_crab1k[i], maxZ=4000, show=False, outfile=None)
                       #outfile=OUTDIR + "image_run328540_evt{}.pdf".format(i))
            """
            evts.append(i)
            pulseheights.append(pulseheight)
            xs.append(x)
            ys.append(y)
            widths.append(width)
            lengths.append(length)
            thetas.append(theta)
            dists.append(dist)
            alphas.append(alpha)
            """
            if args.outfile is not None and not args.flasher:
                with open(ofile, 'a') as paramfileio:
                    paramfileio.write("{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                        i, timestamps[i-current_evt], pulseheight, x, y, width, length, theta, dist, alpha))
        current_evt = stop_evt


    elapsed_time = time.time() - start_time
    print("Elapsed time: {} s".format(elapsed_time))
