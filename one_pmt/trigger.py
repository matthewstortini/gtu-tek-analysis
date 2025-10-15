import pyvisa
import argparse
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

"""
Script written to grab triggers from scope.
Made for using one FEB lane and one PMT.
PMT and FEB output go into scope (so two scope channels)
"""

def init_scope(ip: str, trig_source: str, trig_slope: str, trig_level: float):
    """
    Connect to Tektronix scope and set basic acquisition/trigger parameters.
    Returns the scope object.

    Parameters:
        ip : str
            IP address of the oscilloscope.
        trig_source : str
            Trigger source (e.g. "ch1", "ch2", "ch3", "ch4").
        trig_slope : str
            Trigger slope ("rise" or "fall").
        trig_level : float
            Trigger level in Volts.
    """
    rm = pyvisa.ResourceManager('@py')
    scope = rm.open_resource(f'TCPIP0::{ip}::INSTR')
    scope.timeout = 5000

    # Configure scope
    scope.write('data:encdg ascii')                     # transfer data in ASCII
    scope.write('trigger:a:type edge')                  # edge trigger
    scope.write(f'trigger:a:edge:source {trig_source}') # trigger source
    scope.write(f'trigger:a:level {trig_level:.2f}')    # trigger level
    scope.write(f'trigger:a:edge:slope {trig_slope}')   # rising/falling edge trigger
    scope.write('trigger:a:mode normal')                # wait for valid trigger
    scope.write('acquire:stopafter sequence')           # single acquisition
    scope.write('acquire:state stop')                   # stop until ready to take data

    # print configuration summary for user
    print("=== Scope Configuration ===")
    print(f"IP Address           : {ip}")
    print(f"Data Encoding        : ASCII")
    print(f"Trigger Type         : EDGE")
    print(f"Trigger Source       : {trig_source.upper()}")
    print(f"Trigger Slope        : {trig_slope.upper()}")
    print(f"Trigger Level (V)    : {trig_level:.2f}")
    print(f"Trigger Mode         : NORMAL")
    print(f"Stop After           : SEQUENCE (single acquisition)")
    print(f"Acquisition State    : STOP (waiting to start run)")
    print("============================\n")

    return scope

def get_calibration_constants(scope, channels):
    """
    Query calibration constants for each channel.
    Returns a dictionary of the relevant constants.

    Parameters:
        scope : pyvisa.resources.Resource
            The instrument connection handle.
        channels : list[str]
            List of channel names to query (e.g. ["ch1", "ch3"]).
    """
    calib = {}
    for ch in channels:
        scope.write(f"data:source {ch}")
        calib[ch] = {
            "XINCR": float(scope.query("WFMPRE:XINCR?")),
            "XZERO": float(scope.query("WFMPRE:XZERO?")),
            "XOFF" : float(scope.query("WFMPRE:PT_OFF?")),
            "YMULT": float(scope.query("WFMPRE:YMULT?")),
            "YZERO": float(scope.query("WFMPRE:YZERO?")),
            "YOFF" : float(scope.query("WFMPRE:YOFF?"))
        }

    # print calibration constants for each channel
    print("=== Calibration Constants ===")
    for ch in channels:
        print(f"{ch.upper()} :")
        print(f"   XINCR : {calib[ch]['XINCR']:.6e}")
        print(f"   XZERO : {calib[ch]['XZERO']:.6e}")
        print(f"   XOFF  : {calib[ch]['XOFF']:.3f}")
        print(f"   YMULT : {calib[ch]['YMULT']:.6e}")
        print(f"   YZERO : {calib[ch]['YZERO']:.6e}")
        print(f"   YOFF  : {calib[ch]['YOFF']:.3f}")
    print("==============================\n")

    return calib

def create_run_file(path: str,
                    channels: list[str],
                    wfm_sample_size: int,
                    chunk_size: int = 50,
                    t_dtype: str = "f8",
                    v_dtype: str = "f8",
                    compression: str = "gzip",
                    feb_lane: int = 1,
                    pmt_id: int = 1300
                    ) -> None:

    """
    Create a new HDF5 file for waveform data acquisition.

    Structure:
        /runinfo (group)
            Attributes:
                run_time : float
                    total run time in seconds
                nevents : int
                    number of events in dataset 
                feb_lane : int
                    FEB lane id for this run
                pmt_id : int
                    PMT identifier for this run

        /events (group)
            Filled once per "event", i.e. every time there is a trigger
            Datasets:
                timestamp : float64 [N]
                    timestamp (in seconds) when writing trigger waveforms
                pmt_v : v_dtype [N, wfm_sample_size]
                    Voltage samples for PMT (first scope channel)
                feb_v : v_dtype [N, wfm_sample_size]
                    Voltage samples for FEB (second scope channel)
                channels : string [2]
                    stores the scope channel labels in order (e.g. ["ch1","ch3"])

        /t_axis (group)
            Datasets:
                <chan>_t : t_dtype [wfm_sample_size]
                    Time axis for <chan> -- written once, shared by all events

    Parameters:

       path : str
           Output file path (e.g. "run.h5")
       channels : list[str]
           Scope channels we want to save waveforms for
       wfm_sample_size : int
           Number of samples in each individual waveform
       chunk_size : int
           Number of events to chunk together
       t_dtype : str
           Data type for time samples
       v_dtype : str
           Data type for voltage samples
       compression : str
           Compression algorithm for datasets
       """

    with h5py.File(path, "w") as f:
        
        # define groups
        runinfo = f.create_group("runinfo")
        events = f.create_group("events")
        t_axis = f.create_group("t_axis")

        # run-level attributes
        runinfo.attrs["run_time"] = np.float64(0.0)  # placeholder
        runinfo.attrs["nevents"]  = 0                # placeholder
        runinfo.attrs["feb_lane"] = int(feb_lane)
        runinfo.attrs["pmt_id"] = int(pmt_id)

        # store the scope channel list
        str_dt = h5py.string_dtype(encoding="utf-8")
        events.create_dataset("channels", data=np.asarray(channels, dtype=str_dt))

        # per-event timestamp
        events.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            chunks=(max(1, int(chunk_size)),),
            dtype="f8",
            compression=compression
        )

        # chunking def
        chunk_v = (max(1, int(chunk_size)), int(wfm_sample_size))

        # voltage datasets are fixed names: first=PMT, second=FEB
        events.create_dataset(
            "pmt_v",
            shape=(0, wfm_sample_size),
            maxshape=(None, wfm_sample_size),
            chunks=chunk_v,
            dtype=v_dtype,
            compression=compression
        )
        events.create_dataset(
            "feb_v",
            shape=(0, wfm_sample_size),
            maxshape=(None, wfm_sample_size),
            chunks=chunk_v,
            dtype=v_dtype,
            compression=compression
        )

        # time axis for each scope channel (can differ per channel)
        for ch in channels:
            t_axis.create_dataset(
                f"{ch}_t",
                shape=(wfm_sample_size,),
                dtype=t_dtype,
                compression=compression
            )

def write_waveforms(scope, h5_path, channels, calib_constants, first_waveform):
    """
    Function to save waveforms.

    Parameters:
        scope is the instrument we are communicating with.
        h5_path specifies the file we write to.
        channels is a list that specifies the scope channels to save waveforms for.
        calib_constants needed to convert data to voltage/time values.
        first_waveform tells says if this is the first waveform to be saved.
    """
    # open file for appending
    with h5py.File(h5_path, "a") as f:

        # groups we need to write to
        events = f["events"]
        t_axis = f["t_axis"]

        # current number of events
        n_events = events["timestamp"].shape[0]

        # store timestamp for event
        events["timestamp"].resize((n_events + 1,))
        events["timestamp"][n_events] = float(time.time())

        # loop over scope channels and save waveforms
        for i, ch in enumerate(channels):
            scope.write(f"data:source {ch}")

            # grab waveform and convert comma separated list of strings into an array of floats
            wfm = scope.query("curve?")
            raw = np.fromstring(wfm, sep=",")

            # calibration constants for this scope channel
            YZERO = float(calib_constants[ch]["YZERO"])
            YMULT = float(calib_constants[ch]["YMULT"])
            YOFF  = float(calib_constants[ch]["YOFF"])

            # scale points using calibration constants
            v_points = (raw - YOFF) * YMULT + YZERO

            # pick dataset name by index: 0 -> PMT, 1 -> FEB
            ds_name = "pmt_v" if i == 0 else "feb_v"

            # add one row (new event) for waveform
            n_wfm_samples = events[ds_name].shape[1]
            events[ds_name].resize((n_events + 1, n_wfm_samples))
            events[ds_name][n_events, :] = v_points.astype(events[ds_name].dtype, copy=False)

            # if first waveform, compute and write the per-channel t_axis once
            if first_waveform:
                XZERO = float(calib_constants[ch]["XZERO"])
                XINCR = float(calib_constants[ch]["XINCR"])
                XOFF  = float(calib_constants[ch]["XOFF"])

                # build time axis (seconds) for this scope channel
                t = XZERO + (np.arange(n_wfm_samples) - XOFF) * XINCR

                # write t_axis to hdf5 file
                t_axis[f"{ch}_t"][:] = t.astype(t_axis[f"{ch}_t"].dtype, copy=False)
                
def get_wfm_size(scope, channel):
    """
    Called first time we trigger just to see how many samples are in waveform.
    """
    
    scope.write(f"data:source {channel}")
    wfm = scope.query("curve?")
    raw = np.fromstring(wfm, sep=",")

    n_samples = raw.size

    return n_samples

def write_end_run(h5_path, elapsed_time, n_triggers):
    """
    Function to finalize run and write summary attributes.

    Parameters:
        h5_path : str
            Path to HDF5 file being written.
        elapsed_time : float
            Total run time in seconds.
        n_triggers : int
            Total number of triggers (events) recorded in this run.
    """
    if n_triggers == 0:
        return
    
    with h5py.File(h5_path, "a") as f:

        # groups we need to write to
        runinfo = f["runinfo"]

        # update attributes
        runinfo.attrs["run_time"] = np.float64(elapsed_time)
        runinfo.attrs["nevents"]  = int(n_triggers)

def make_live_plots(h5_path, channels, plot_scales, plot_labels):
    """
    Function to make live plots as data is taken

    Produces three figures:
      (1) Most recent waveform (overlay all scope channels)
      (2) Last 25 waveforms (overlay all scope channels)
      (3) Last 3 events (overlay all scope channels per event; one color per event)

    Reads from HDF5 file

    Parameters:
        h5_path : str
            Path to HDF5 file being written/read.
        channels : list[str]
            Scope channels to plot (order must match datasets in file).
        plot_scales : list[float]
            Per-channel multiplicative scale to apply to voltages when plotting
            (one entry per scope channel).
        plot_labels : list[str]
            Per-channel legend labels (one entry per scope channel).
    """
    # colors
    color_cycle = ["black", "blue"]
    ch_color = {}
    for i, ch in enumerate(channels):
        ch_color[ch] = color_cycle[i]

    # open file, grab most recent waveform and last 25 (if we have 25 yet)
    with h5py.File(h5_path, "r") as f:
        events = f["events"]
        t_axis = f["t_axis"]

        n_events = events["timestamp"].shape[0]
        if n_events == 0:
            return

        # --------------------------
        # (1) Most recent waveform
        # --------------------------
        plt.figure(num=1)
        plt.clf()

        last_idx = n_events - 1
        for i, ch in enumerate(channels):
            # time stored in seconds -> convert to ns for plotting
            t = np.asarray(t_axis[f"{ch}_t"][:], dtype=np.float64) * 1e9
            # voltages from fixed datasets
            ds_name = "pmt_v" if i == 0 else "feb_v"
            v = np.asarray(events[ds_name][last_idx, :], dtype=np.float64)

            # apply per-channel scale and label
            v_plot = v * float(plot_scales[i])
            lbl = plot_labels[i]

            plt.plot(t, v_plot, color=ch_color[ch], label=lbl)

        plt.title("Most recent waveform", loc="left")
        plt.xlabel("time (ns)", ha="right", x=1.0)
        plt.ylabel("voltage (scaled)", ha="right", y=1.0)
        plt.xlim(-50.0, 50.0)
        plt.legend(frameon=True, loc="upper left", fontsize="medium")
        plt.grid(True)
        plt.tight_layout()

        # --------------------------
        # (2) Last 25 waveforms
        # --------------------------
        plt.figure(num=2)
        plt.clf()

        start_idx = max(0, n_events - 25)
        labeled = {ch: False for ch in channels} # track what channels have been labeled

        for idx in range(start_idx, n_events):
            for i, ch in enumerate(channels):
                t = np.asarray(t_axis[f"{ch}_t"][:], dtype=np.float64) * 1e9
                ds_name = "pmt_v" if i == 0 else "feb_v"
                v = np.asarray(events[ds_name][idx, :], dtype=np.float64)

                # apply per-channel scale
                v_plot = v * float(plot_scales[i])

                # label each channel first time it is drawn
                if not labeled[ch]:
                    lbl = plot_labels[i]
                    labeled[ch] = True
                else:
                    lbl = None

                plt.plot(t, v_plot, color=ch_color[ch], label=lbl, linewidth=1.0)

        plt.title("Last (up to) 25 events", loc="left")
        plt.xlabel("time (ns)", ha="right", x=1.0)
        plt.ylabel("voltage (scaled)", ha="right", y=1.0)
        plt.xlim(-50.0, 50.0)
        plt.legend(frameon=True, loc="upper left", fontsize="medium")
        plt.grid(True)
        plt.tight_layout()

        # -----------------------------------------------
        # (3) Last 3 events, per-event color/legend entry
        # -----------------------------------------------
        plt.figure(num=3)
        plt.clf()

        # choose (up to) the last 3 events
        n_show = min(3, n_events)
        first_idx = n_events - n_show

        # colors for the three events we draw
        event_colors = ["black", "blue", "purple"]

        for j, idx in enumerate(range(first_idx, n_events)):
            evt_color = event_colors[j]  

            # build a readable legend label for the event
            evt_label = f"event {idx + 1}"

            # draw all channels for this event with the same color
            for i, ch in enumerate(channels):
                t = np.asarray(t_axis[f"{ch}_t"][:], dtype=np.float64) * 1e9
                ds_name = "pmt_v" if i == 0 else "feb_v"
                v = np.asarray(events[ds_name][idx, :], dtype=np.float64)

                v_plot = v * float(plot_scales[i])

                # label only once per event so legend shows up to 3 entries (one per event)
                lbl = evt_label if i == 0 else None
                plt.plot(t, v_plot, color=evt_color, label=lbl, linewidth=1.2)

        plt.title("Last (up to) 3 events (all channels)", loc="left")
        plt.xlabel("time (ns)", ha="right", x=1.0)
        plt.ylabel("voltage (scaled)", ha="right", y=1.0)
        plt.xlim(-50.0, 50.0)
        plt.legend(frameon=True, loc="upper left", fontsize="medium")
        plt.grid(True)
        plt.tight_layout()

    plt.pause(0.001)


def main():

    # add argument parser, define default values, print out settings for user
    parser = argparse.ArgumentParser(
        description="Save waveforms from Tektronix scope to HDF5 every time the scope triggers."
    )

    parser.add_argument(
        "--ip-address", type=str, default="192.168.1.50",
        help="IP address of the oscilloscope"
    )
    parser.add_argument(
        "--channels", nargs="+", default=["ch1", "ch3"],
        help="Specify two channels on scope (first one for PMT and second for FEB) (e.g. --channels ch1 ch2)"
    )
    parser.add_argument(
        "--pmt-id", type=int, default=1300,
        help="PMT identifier (integer)"
    )
    parser.add_argument(
        "--feb-lane", type=int, default=1,
        help="FEB lane id (integer)"
    )
    parser.add_argument(
        "--output", type=str, default="run.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--max-time", type=int, default=200,
        help="Maximum run time in seconds"
    )
    parser.add_argument(
        "--max-events", type=int, default=5,
        help="Maximum number of triggered events before ending run"
    )
    parser.add_argument(
        "--live-plots", action="store_true",
        help="Enable live plotting during acquisition"
    )
    parser.add_argument(
        "--plot-scales", nargs="+", type=float, default=None,
        help="Per-channel y-scale multipliers for plotting, one per channel (e.g. --plot-scales 10 0.5)"
    )
    parser.add_argument(
        "--plot-label", dest="plot_labels", action="append", default=None,
        help="Legend label for a channel; repeat per channel (e.g. --plot-label 'PMT (100 mV/div)' --plot-label 'FEB (2.0 V/div)')"
    )
    parser.add_argument(
        "--trigger-source", type=str, default="ch3",
        help="Trigger source channel (e.g. ch1, ch2, ch3, ch4)"
    )
    parser.add_argument(
        "--trigger-slope", type=str, choices=["rise", "fall"], default="rise",
        help="Trigger edge slope (rise or fall)"
    )
    parser.add_argument(
        "--trigger-level", type=float, default=2.0,
        help="Trigger level in Volts (e.g. 2.0)"
    )
    
    # parse and initialize command line arguments
    args = parser.parse_args()    
    ip = args.ip_address
    channels = args.channels
    pmt_id  = args.pmt_id
    feb_lane = args.feb_lane
    h5_path = args.output
    maxtime = args.max_time
    maxevents = args.max_events
    makeliveplots = args.live_plots
    plot_scales = args.plot_scales # None or list[float]
    plot_labels = args.plot_labels # None or list[str]
    trigger_source = args.trigger_source
    trigger_slope  = args.trigger_slope
    trigger_level  = args.trigger_level

    # check we have two scope channels
    if len(channels) != 2:
        print("You did not provide the correct number of scope channels, script configured for two ... exiting")
        return

    # make sure plot scales and plot labels (if given) match the number of channels
    if plot_scales is not None and len(plot_scales) != len(channels):
        print(f"Warning: provided {len(plot_scales)} plot scales but {len(channels)} channels; ignoring plot scales.")
        plot_scales = None
    if plot_labels is not None and len(plot_labels) != len(channels):
        print(f"Warning: provided {len(plot_labels)} plot labels but {len(channels)} channels; ignoring plot labels.")
        plot_labels = None

    # define plotting defaults if not given but live plotting mode is on
    if makeliveplots:
        if plot_scales is None:
            plot_scales = [1.0] * len(channels)
        if plot_labels is None:
            plot_labels = ["PMT", "FEB"]
        plt.ion()

    # print configuration for users
    print("\n=== Run Configuration ===")
    print(f"Scope IP Address  : {ip}")
    print(f"Scope Channels    : {channels}")
    print(f"PMT ID            : {pmt_id}")
    print(f"FEB Lane          : {feb_lane}")
    print(f"Output File       : {h5_path}")
    print(f"Max Time (s)      : {maxtime}")
    print(f"Max Events        : {maxevents}")
    print(f"Live Plotting     : {makeliveplots}")
    if makeliveplots:
        print(f"Plot Scales       : {plot_scales}")
        print(f"Plot Labels       : {plot_labels}")
    print("==========================\n")

    # initialize/configure scope and get calibration constants
    scope = init_scope(ip, trigger_source, trigger_slope, trigger_level)
    calibrations = get_calibration_constants(scope, channels)

    # start run and note start time
    print("BEGINNING RUN\n")
    scope.write('acquire:state run')
    start_time = time.time()

    # initialize runtime control variables
    n_triggers = 0
    continue_run = True
    caught_interrupt = False
    next_time_update = maxtime / 10.0  # next time to print 10% progress
    percent_done = 10

    while continue_run:
        try:
            elapsed_time = time.time() - start_time

            # check if we should print elapsed time progress
            if elapsed_time >= next_time_update and percent_done <= 100:
                print(f"{percent_done}% time elapsed ({elapsed_time:.1f} s)")
                percent_done += 10
                next_time_update += maxtime / 10.0

            # end run if we have reached end or KeyboardInterrupt was detected
            if (n_triggers >= maxevents) or (elapsed_time >= maxtime) or caught_interrupt:
                write_end_run(h5_path, elapsed_time, n_triggers)
                print(f"\nRun finalized: {n_triggers} events saved, elapsed time {elapsed_time:.1f} s.")
                continue_run = False
                break

            # main acquisition logic -- checking for state change
            if int(scope.query('acquire:state?')) == 0:
                if n_triggers == 0:
                    wfm_size = get_wfm_size(scope, channels[0])
                    create_run_file(h5_path, channels, wfm_size,
                                    feb_lane=feb_lane, pmt_id=pmt_id)
                    write_waveforms(scope, h5_path, channels, calibrations, True)
                    n_triggers += 1
                    print(f"FOUND AND SAVED {n_triggers}ST TRIGGER")
                    scope.write('acquire:state run')
                else:
                    write_waveforms(scope, h5_path, channels, calibrations, False)
                    n_triggers += 1
                    suffix = {1: "ST", 2: "ND", 3: "RD"}.get(n_triggers if n_triggers < 20 else n_triggers % 10, "TH")
                    print(f"FOUND AND SAVED {n_triggers}{suffix} TRIGGER")
                    scope.write('acquire:state run')        
            else:
                time.sleep(0.01) # pause for 10 ms if still in run state

            # make plots if desired
            if makeliveplots and n_triggers > 0:
                make_live_plots(h5_path, channels, plot_scales, plot_labels)

        except KeyboardInterrupt:
            caught_interrupt = True
            print("\nKeyboardInterrupt detected... finalizing run")

            
if __name__ == "__main__":
    main()
