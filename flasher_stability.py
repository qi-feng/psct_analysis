from pSCT_analysis import *
from scipy.signal import medfilt2d
import time

DATADIR='/mnt/data476G/pSCT_data/'
OUTDIR='./'
norm_map_default = np.load("norm_map_default.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pSCT analysis')
    #parser.add_argument('run', type=int, default=328540, help="Run number")
    parser.add_argument('-l', '--runlist', default=None, nargs='+', help='List of run ids')
    parser.add_argument('-i', '--inlist', default=None, nargs='+', help='Input file list')
    #parser.add_argument('-i', '--infile', default=None, help="Read calibrated file.")
    parser.add_argument('--peak_ADC_lower', type=int, default=1400, help="Cut on peak ADC lower; anything with peak below this value is thrown away")
    #parser.add_argument('-i', '--interactive', action="store_true", help="Flag to show interactive plots.")
    parser.add_argument('-s', '--save', action="store_true", help="Flag to save plots.")
    parser.add_argument('--flatfield', action="store_true", help="Flag to try flatfielding.")
    parser.add_argument('-f', '--flasher_file', default="", help="File to save flasher event numbers.")
    parser.add_argument('--outfile', default=None, help="Text file to save parameters to. ")
    parser.add_argument('--outdir', default=None, help="Default to current dir ")
    parser.add_argument('--datadir', default=None, help="Default to dir {}".format(DATADIR))
    parser.add_argument('--smooth', action="store_true", help="Show/save the smoothed image (using a 3x3 median kernel).")

    args = parser.parse_args()

    run_nums = args.runlist
    infiles = args.inlist

    if args.datadir is not None:
        DATADIR = args.datadir
    if args.outdir is not None:
        OUTDIR = args.outdir


    if args.flasher_file=="":
        flasher_file = OUTDIR +"/flasher_res_runs{}.npy".format("_".join(run_nums))
    else:
        flasher_file = OUTDIR + args.flasher_file
    with open(flasher_file, 'w') as ffio:
        ffio.write("run_num,evt_start,evt_stop,t_start,t_stop,monitor_charge,monitor_charge_err\n")

    colnames = ['run_num', 'evt_num', 'timestamp', 'median_charge', 'std_charge']
    ofile = OUTDIR + "/" + args.outfile
    with open(ofile, 'w') as paramfileio:
            paramfileio.write(" ".join(colnames))
            paramfileio.write("\n")

    n_evt_per_read = 1800
    n_read = 3
    start_time = time.time()

    for run_num, file_ in zip(args.runlist, args.inlist):
        filename = "{}/{}".format(DATADIR, file_)
        reader = target_io.WaveformArrayReader(filename)

        isR1 = reader.fR1
        n_total_events = reader.fNEvents

        print("Reading events in run {}; found {} events in total".format( run_num, n_total_events ))

        read_per_cycle = 3000
        start_evts = [ i*(n_total_events//(n_read)) for i in range(n_read)]


        # ampl_crab5k, blocks_crab5k, phases_crab5k = read_raw_signal(reader_crab, range(5000))

        evts = []
        ts = []
        meds = []
        stds = []

        current_evt = 0

        for iread in range(n_read):

            current_evt = start_evts[iread]
            n_flasher_read = 0
            read_finished = False

            # in case 1 read doesn't give needed number of flasher events

            for icycle in range(10):
                if read_finished:
                    break
                elif current_evt == n_total_events:
                    print("reached the end")
                    continue
                else:
                    stop_evt = min(current_evt + read_per_cycle, n_total_events)


                print("Reading evt {} to {}...".format(current_evt, stop_evt-1))
                #timestamps, ampl, blocks, phases = read_raw_signal(reader, range(current_evt, stop_evt), get_timestamp=True, calibrated = calibrated)
                #timestamps, ampl, blocks, phases = read_raw_signal_array(reader, range(current_evt, stop_evt), get_timestamp=True, calibrated = calibrated)
                ampl, timestamps, first_cell_ids, stale_bit = read_calibrated_data(file_, DATADIR=DATADIR,
                                                                                        event_list=range(current_evt, stop_evt))

                for i in range(current_evt, stop_evt):
                    im = show_image(ampl[i-current_evt], maxZ=4000, show=False)
                    im_smooth = medfilt2d(im, 3)

                    #if np.percentile(im_smooth[im_smooth != 0], 90) > 500:
                    if np.percentile(im_smooth[im_smooth != 0], 20) < 200:
                        #print("This is probably a flasher event")
                        #n_flasher_read += 1
                    #else: #not a flasher
                        continue

                    if args.flatfield:
                        im = im / norm_map_default
                        #if args.cleaning:
                        #    im_clean = cleaning(im)
                        #    im_smooth = medfilt2d(im_clean, 3)
                        #else:
                        im_smooth = medfilt2d(im, 3)

                    #elif args.cleaning:
                    #    im_clean = cleaning(im)
                    #    im_smooth = medfilt2d(im_clean, 3)
                    else:
                        im_smooth = medfilt2d(im, 3)

                    if (im_smooth>50).sum() < 10:
                        continue

                    n_flasher_read += 1
                    evts.append(i)
                    ts.append(timestamps[i-current_evt])


                    if args.smooth:
                        meds.append(np.median(im_smooth))
                        stds.append(np.std(im_smooth))
                    else:
                        #ignoring cleaning
                        meds.append(np.median(im))
                        stds.append(np.std(im))

                    with open(ofile, 'a') as ffio:
                        ffio.write("{},{},{},{},{}\n".format(run_num,i,timestamps[i-current_evt],meds[-1],stds[-1]))

                    if args.save:
                        if args.smooth:
                                fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                                plt.pcolor(im_smooth, cmap=plt.cm.gray)
                                plt.xlim(0, 40)
                                plt.ylim(0, 40)
                                plt.colorbar()
                                plt.tight_layout()
                                plt.savefig(OUTDIR +"/flasher_calibrated_smooth_image_run{}_evt{}.png".format(run_num, i))
                        else:
                                fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
                                plt.pcolor(im, cmap=plt.cm.gray)
                                plt.xlim(0, 40)
                                plt.ylim(0, 40)
                                plt.colorbar()
                                plt.tight_layout()
                                plt.savefig(OUTDIR +"/flasher_calibrated_image_run{}_evt{}.png".format(run_num, i))

                        np.save(OUTDIR +"/flasher_ampl_run{}_evt{}_firstcell{}.npy".format(run_num, i, first_cell_ids[i - current_evt]), ampl[i - current_evt])
                        np.save(OUTDIR +"/flasher_im_run{}_evt{}_firstcell{}.npy".format(run_num, i, first_cell_ids[i - current_evt]), im)
                        #show_image(ampl_crab1k[i], maxZ=4000, show=False, outfile=None)
                        #outfile=OUTDIR + "image_run328540_evt{}.pdf".format(i))

                    if n_flasher_read == n_evt_per_read:
                        #finished this cycle
                        read_finished = True
                        break

                current_evt = stop_evt

            with open(flasher_file, 'a') as ffio:
                #ffio.write("run_num,evt_start,evt_stop,t_start,t_stop,monitor_charge,monitor_charge_err\n")
                ffio.write("{},{},{},{},{},{},{}\n".format(run_num,np.min(evts),np.max(evts),np.min(ts),np.max(ts),np.mean(meds),np.std(meds)))

    elapsed_time = time.time() - start_time
    print("Elapsed time: {} s".format(elapsed_time))
