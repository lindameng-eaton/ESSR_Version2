import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict

from scipy.signal import medfilt

from HiZLib.utility.others import substring_exists_in_list

# user parameters
TIME_COLUMN = "Time"

# internal parameters:
VOLTAGE_PREFIX = "V"
CURRENT_PREFIX = "I"
NUM_PHASES = 3
STATUS_COLUMN = "Status"
OFF_STATUS = 0
ON_STATUS = 1
CSV_EXT = ".csv"
HDF_EXT = ".h5"
HDF_KEY = "/Dataset"
MEDIAN_FILTER_KERNEL_SIZE = 99
BRKR_STATUS_THRESHOLD = 0.2
SPECIFIED_CH_KEY = "spec"

Channels = Dict[str, List[str]]


def data_acquisition(
    file_path: str,
    voltage_channels: List[str] = [],
    current_channels: List[str] = [],
    nodes: List[str] = [],
    kwds_brkr: List[str] = ["brk"],
) -> Tuple[pd.DataFrame, Channels, Channels]:
    """Reads data from file and outputs a dataframe comprising time, voltage and current of user specified channels,
     and breaker status together with voltage and current channel dictionaries

    file_path: path of file to retrieve data from
    v_channels: user specified voltage channels
    i_channels: user specified current channels
    nodes: user specified nodes; from each node voltage/current channel-names (3 each) are automatically derived
    by adding prefix "V" or "I" for voltage and current and suffix "1", "2", or "3" for phases A, B, or C, respectively.
    Ex: If NODE_NAME = "pt3", derived voltage and current channels will be ["Vpt31", "Vpt32", "Vpt33"] and
    ["Ipt31", "Ipt32", "Ipt33"] respectively.
    kwds_brks: user specified keywords for breaker"""
    data = read_data(file_path)
    (
        all_available_channels,
        available_voltage_channels,
        available_current_channels,
    ) = get_available_channel_names(
        data.columns,
        voltage_channels=voltage_channels,
        current_channels=current_channels,
        nodes=nodes,
    )
    breaker_status = estimate_breaker_status(data, kwds_brkr=kwds_brkr)
    return (
        pd.concat(
            [
                data[TIME_COLUMN],
                data.loc[:, all_available_channels],
                breaker_status,
            ],
            axis=1,
        ),
        available_voltage_channels,
        available_current_channels,
    )


def read_data(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith(CSV_EXT):
        data = pd.read_csv(file_path)
    elif file_path.lower().endswith(HDF_EXT):
        with pd.HDFStore(file_path, "r") as hdf:
            data = hdf[HDF_KEY]
    else:
        raise OSError(
            f"Unrecognized data format ... only accepts data in {HDF_EXT} and {CSV_EXT}!!!"
        )
    return data


def get_available_channel_names(
    data_columns: pd.Index,
    current_channels: list,
    voltage_channels: list,
    nodes: List[str],
) -> Tuple[List[str], Channels, Channels]:
    """Returns list of voltage and current channels available in 'all_columns'"""

    def available_channels(all_channels: List[str]) -> List[str]:
        return [col for col in data_columns if col in all_channels]

    def get_available_channels_from_node(
        node: str,
    ) -> Tuple[List[str], List[str]]:
        """Derives list of voltage and current channel names from node.
        Ex: If node = "pt3", derived voltage and current channels will be
        ["Vpt31", "Vpt32", "Vpt33"] and ["Ipt31", "Ipt32", "Ipt33"] respectively"""
        v_ch = [VOLTAGE_PREFIX + node + str(ph + 1) for ph in range(NUM_PHASES)]
        curr_ch = [CURRENT_PREFIX + node + str(ph + 1) for ph in range(NUM_PHASES)]
        return available_channels(v_ch), available_channels(curr_ch)

    all_channels, available_v_ch, available_curr_ch = [], {}, {}
    for node in nodes:
        v_ch, curr_ch = get_available_channels_from_node(node)
        if len(v_ch) + len(curr_ch) > 0:
            all_channels, available_v_ch[node], available_curr_ch[node] = (
                all_channels + v_ch + curr_ch,
                v_ch,
                curr_ch,
            )
    if len(voltage_channels) + len(current_channels) > 0:
        available_v_ch_spec = available_channels(voltage_channels)
        available_curr_ch_spec = available_channels(current_channels)
        (
            all_channels,
            available_v_ch[SPECIFIED_CH_KEY],
            available_curr_ch[SPECIFIED_CH_KEY],
        ) = (
            all_channels + available_v_ch_spec + available_curr_ch_spec,
            available_v_ch_spec,
            available_curr_ch_spec,
        )

    return all_channels, available_v_ch, available_curr_ch


def estimate_breaker_status(
    data: pd.DataFrame,
    kwds_brkr: Union[List[str], str],
    kernel_size: int = MEDIAN_FILTER_KERNEL_SIZE,
    threshold: float = BRKR_STATUS_THRESHOLD,
) -> Optional[pd.DataFrame]:
    """Returns a single column DataFrame with series of estimated breaker status"""
    brkr_signal_columns = [
        col for col in data.columns if substring_exists_in_list(kwds_brkr, col)
    ]
    if len(brkr_signal_columns) > 0:
        for col in brkr_signal_columns:
            data.loc[:, col] = np.abs(data.loc[:, col] - data.loc[:, col][0])
        # sum up all the breaker signal columns
        status = data.loc[:, brkr_signal_columns].sum(axis=1).to_frame(STATUS_COLUMN)
        # differential value of median filtered version of summed up breaker signals
        diff_filtered_values = np.diff(
            medfilt(status.values.flatten(), kernel_size=kernel_size)
        )

        # estimate breaker status change indices
        indices_on = np.argwhere(diff_filtered_values > threshold)
        indices_off = np.argwhere(diff_filtered_values < -threshold)
        status.iloc[:] = OFF_STATUS
        if len(indices_on):
            status.iloc[indices_on[0][0] :] = ON_STATUS
        if len(indices_off):
            status.iloc[indices_off[0][0] :] = OFF_STATUS
    else:
        status = None
    return status
