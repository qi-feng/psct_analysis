from pSCT_analysis import *
from scipy.signal import medfilt2d

#DATADIR='/a/data/tehanu/pSCTdata'
DATADIR='/mnt/data476G/pSCT_data/'
OUTDIR='./'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('evt', type=int, default=7, help="Start event number")
    parser.add_argument('-t', '--trace', action="store_true", help="Plot traces")
    parser.add_argument('-c', '--calibrated', default=None, help="Read calibrated file instead of raw file.")
    parser.add_argument('-b', '--baseline', action="store_true", help="Subtract baselines from the first 15 samples and the last 20 samples in traces")
    #parser.add_argument('-n', '--num_evt', type=int, default=1, help="Number of events to read.Default is 1.")
    parser.add_argument('--outdir', default=None, help="Default to current dir ")
    parser.add_argument('--datadir', default=None, help="Default to dir {}".format(DATADIR))
    args = parser.parse_args()


    if args.datadir is not None:
        DATADIR = args.datadir
    if args.outdir is not None:
        OUTDIR = args.outdir

    #example just to read 10 evts and plot one
    run_num = args.run
    evt_num = args.evt
    #n_evts = int(args.num_evt)
    n_evts = 1
    print("Reding {} events starting from evt {} in run {}".format(n_evts, evt_num, run_num))

    if args.calibrated is not None:
        reader = get_reader_calibrated(args.calibrated, DATADIR=DATADIR) #("cal328555_100evt.r1")
        calib_string = "calibrated_pedHVoff_no_baseline_subtraction"
    else:
        reader = get_reader(run_num)
        if args.baseline:
            calib_string = "simple_baseline_subtraction"
        else:
            calib_string = "raw_no_baseline_subtraction"
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    ampl, blocks, phases = read_raw_signal(reader, range(evt_num, evt_num+n_evts))
    #print("1")
    #print(ampl.shape, ampl[0].shape, blocks[0], phases[0])
    #print(ampl[0])
    #im = show_image(ampl[0], minZ=-1000, maxZ=2500, show=True, outfile=OUTDIR + "im_{}_run{}_evt{}".format(calib_string, run_num, evt_num))
    im = show_image(ampl[0], minZ=-500, maxZ=2000, show=True, outfile=OUTDIR + "im_{}_run{}_evt{}".format(calib_string, run_num, evt_num))

    #print("2")
    im_smooth = medfilt2d(im, 3)
    print("90 percentile ADC: ", np.percentile(im_smooth[im_smooth != 0], 90))
    if np.percentile(im_smooth[im_smooth!=0], 90)>500:
        print("This is probably a flasher event")
        isf = 'f'
    else:
        isf = ''

    np.save(OUTDIR + "/ampl_{}_run{}_evt{}_block{}_phase{}{}.npy".format(calib_string, run_num, evt_num, int(blocks[0]),
                                                                    int(phases[0]), isf), ampl[0])
    np.save(OUTDIR + "/im_{}_run{}_evt{}_block{}_phase{}{}.npy".format(calib_string, run_num, evt_num, int(blocks[0]),
                                                                  int(phases[0]), isf), im)

    if args.trace:
        if args.baseline:
            plot_traces(ampl, 0, mods=range(nModules), asics = range(nasic), channels=range(nchannel),
                blocks=blocks, phases=phases,
                ylim=[-1000,1500],
                show=True, out_prefix="traces_{}_{}_evt{}".format(calib_string, run_num, evt_num), interactive=False)
        else:
            plot_traces(ampl, 0, mods=range(nModules), asics=range(nasic), channels=range(nchannel),
                        #blocks=blocks, phases=phases,
                        #ylim=[-1000, 1500],
                        #ylim=[-100, 3500],
                        ylim=[-500, 2000],
                        show=True, out_prefix="traces_{}_{}_evt{}".format(calib_string, run_num, evt_num),
                        interactive=False)
