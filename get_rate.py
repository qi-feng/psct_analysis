from pSCT_analysis import *
import time

DATADIR='/mnt/data476G/pSCT_data/'
OUTDIR='./'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    parser.add_argument('-s', '--save', action="store_true", help="Flag to save plots.")
    parser.add_argument('-v', '--verbose', action="store_true", help="Flag to be verbose.")
    parser.add_argument('-b','--binsize', type=float, default=1.0, help="Bin size in s. ")
    parser.add_argument('-o','--outfile', default=None, help="Text file to save parameters to. ")
    parser.add_argument('--outdir', default=None, help="Default to current dir ")
    parser.add_argument('--datadir', default=None, help="Default to dir {}".format(DATADIR))
    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run
    if args.datadir is not None:
        DATADIR = args.datadir
    if args.outdir is not None:
        OUTDIR = args.outdir
    show = args.interactive
    if not args.interactive and not args.save:
        print("You didn't specify either interactive or save. Assume interactive. ")
        show=True
    reader = get_reader(run_num, DATADIR=DATADIR)
    n_evts = reader.GetNEvents()
    evt_start = 0

    if args.outfile is not None:
        colnames = ['evt_num', 'timestamp']
        ofile = OUTDIR + "/" + args.outfile
        with open(ofile, 'w') as paramfileio:
            paramfileio.write(" ".join(colnames))
            paramfileio.write("\n")

    print("Reading {} events starting from evt {} in run {}".format(n_evts, n_evts, run_num))
    start_time = time.time()

    read_per_cycle = 1000
    ncycles = n_evts//read_per_cycle + 1

    evts = np.zeros(n_evts)
    timestamps= np.zeros(n_evts)
    if evts.nbytes/1e9 >4:
        print("Warning will use {} GB memory".format(evts.nbytes/1e9*2))
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    current_evt = evt_start
    for icycle in range(ncycles):
        if icycle == (ncycles - 1):
            stop_evt = n_evts
        else:
            stop_evt = current_evt + read_per_cycle
        if args.verbose:
            print("Reading evt {} to {}...".format(current_evt, stop_evt))
        evts[current_evt:stop_evt], timestamps[current_evt:stop_evt] = read_timestamps(reader, range(current_evt, stop_evt), verbose=args.verbose)

        if args.outfile is not None:
            with open(ofile, 'a') as paramfileio:
                for evt_, t_ in zip(evts[current_evt:stop_evt], timestamps[current_evt:stop_evt] ):
                    paramfileio.write("{} {}\n".format(evt_, t_))
        current_evt = stop_evt


    fig = plt.figure()
    ax = fig.add_subplot(111)
    timestamps = timestamps/1e9
    plt.hist(timestamps, bins=np.arange(timestamps[0], np.max(timestamps), args.binsize))
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:.0f}'.format(x / args.binsize) for x in y_vals])
    plt.xlabel("Timestamps (s)")
    plt.ylabel("Rate (Hz)")
    plt.legend(loc='best')
    plt.tight_layout()
    if args.outfile is None:
        outfile = "rate_run{}_bin{:.0f}s.png".format(run_num, args.binsize)
    else:
        outfile = args.outfile[:-4]+".png"
    plt.savefig(outfile)

