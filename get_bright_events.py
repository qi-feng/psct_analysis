from pSCT_analysis import *
from scipy.signal import medfilt2d
import time

DATADIR='/mnt/data476G/pSCT_data/'
OUTDIR='./'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('--start_evt', type=int, default=0, help="Start event number")
    parser.add_argument('-n', '--num_evt', type=int, default=-1, help="Number of events to read. Default is all events.")
    parser.add_argument('--pead_ADC_lower', type=int, default=1400, help="Cut on peak ADC lower; anything with peak below this value is thrown away")
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    parser.add_argument('-s', '--save', action="store_true", help="Flag to save plots.")
    parser.add_argument('--smooth', action="store_true", help="Show/save the smoothed image (using a 3x3 median kernel).")
    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run
    evt_start = args.start_evt
    n_evts = int(args.num_evt)
    show = args.interactive
    if not args.interactive and not args.save:
        print("You didn't specify either interactive or save. Assume interactive. ")
        show=True
    reader = get_reader(run_num)
    if n_evts == -1:
        n_evts = reader.GetNEvents()

    print("Reding {} events starting from evt {} in run {}".format(n_evts, n_evts, run_num))
    start_time = time.time()

    read_per_cycle = 100
    ncycles = n_evts//read_per_cycle + 1
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    current_evt = evt_start
    for icycle in range(ncycles):
        if icycle == (ncycles - 1):
            stop_evt = n_evts
        else:
            stop_evt = current_evt + read_per_cycle
        print("Reading evt {} to {}...".format(current_evt, stop_evt))
        ampl, blocks, phases = read_raw_signal(reader, range(current_evt, stop_evt))

        for i in range(current_evt, stop_evt):
            im = show_image(ampl[i-current_evt], maxZ=4000, show=False)
            im_smooth = medfilt2d(im, 3)
            if np.max(im_smooth) < args.pead_ADC_lower:
                continue
            #plt.figure()
            #ax = plt.subplot(111)
            #cx = plt.pcolor(im_smooth, vmin=1, vmax=4000)
            if args.save:
                if args.smooth:
                    fit_gaussian2d(im_smooth, outfile="smoothed_image_run{}_evt{}.png".format(run_num, i))
                else:
                    fit_gaussian2d(im, outfile="image_run{}_evt{}.png".format(run_num, i))
                if show:
                    plt.colorbar()
                    plt.show()
            else:
                if args.smooth:
                    fit_gaussian2d(im_smooth)
                else:
                    fit_gaussian2d(im)
                plt.colorbar()
                plt.show()
            #show_image(ampl_crab1k[i], maxZ=4000, show=False, outfile=None)
                       #outfile=OUTDIR + "image_run328540_evt{}.pdf".format(i))
        current_evt = stop_evt
    elapsed_time = time.time() - start_time
    print("Elapsed time: {} s".format(elapsed_time))
