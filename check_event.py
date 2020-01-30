from pSCT_analysis import *
from scipy.signal import medfilt2d

#DATADIR='/a/data/tehanu/pSCTdata'
DATADIR='/mnt/data476G/pSCT_data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('evt', type=int, default=7, help="Start event number")
    parser.add_argument('-t', '--trace', action="store_true", help="Plot traces")
    #parser.add_argument('-n', '--num_evt', type=int, default=1, help="Number of events to read.Default is 1.")
    args = parser.parse_args()

    #example just to read 10 evts and plot one
    run_num = args.run
    evt_num = args.evt
    #n_evts = int(args.num_evt)
    n_evts = 1
    print("Reding {} events starting from evt {} in run {}".format(n_evts, evt_num, run_num))

    reader = get_reader(run_num)
    # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

    ampl, blocks, phases = read_raw_signal(reader, range(evt_num, evt_num+n_evts))

    im = show_image(ampl[0], maxZ=3500, show=True, outfile=OUTDIR + "im_run{}_evt{}".format(run_num, evt_num))

    im_smooth = medfilt2d(im, 3)
    print("90 percentile ADC: ", np.percentile(im_smooth[im_smooth != 0], 90))
    if np.percentile(im_smooth[im_smooth!=0], 90)>500:
        print("This is probably a flasher event")
        isf = 'f'
    else:
        isf = ''

    np.save(OUTDIR + "/ampl_run{}_evt{}_block{}_phase{}{}.npy".format(run_num, evt_num, int(blocks[0]),
                                                                    int(phases[0]), isf), ampl[0])
    np.save(OUTDIR + "/im_run{}_evt{}_block{}_phase{}{}.npy".format(run_num, evt_num, int(blocks[0]),
                                                                  int(phases[0]), isf), im)

    if args.trace:
        plot_traces(ampl, 0, mods=range(nModules), asics = range(nasic), channels=range(nchannel),
                blocks=blocks, phases=phases,
                ylim=[-100,1500],
                show=True, out_prefix="traces_{}_evt{}".format(run_num, evt_num), interactive=False)