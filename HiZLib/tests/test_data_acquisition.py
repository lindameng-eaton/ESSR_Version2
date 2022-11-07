import pytest
import pandas as pd
from HiZLib.data_preprocessing.data_acquisition import (
    data_acquisition,
    get_available_channel_names,
    estimate_breaker_status,
    STATUS_COLUMN,
    ON_STATUS,
    OFF_STATUS,
    SPECIFIED_CH_KEY,
)
import os


@pytest.fixture
def volt_channels():
    return ["channel_1"]


@pytest.fixture
def curr_channels():
    return ["channel_3", "channel_4"]


@pytest.fixture
def node_name():
    return ["pt4", "pt3"]


@pytest.fixture
def breaker_keywords():
    return ["solenoid", "Trigger", "brkr"]


class TestChannelNames:
    @pytest.fixture
    def dataframe_node_based_channels(self):
        return pd.DataFrame(
            {
                "other": [-1, -2, -3],
                "Vpt41": [1, 2, 3],
                "Vpt42": [4, 5, 6],
                "Vpt32": [4, 5, 6],
                "Ipt41": [7, 8, 9],
                "Ipt42": [10, 11, 12],
                "Ipt43": [13, 14, 15],
            }
        )

    @pytest.fixture
    def dataframe_specified_channels(self):
        return pd.DataFrame(
            {
                "other": [-1, -2, -3],
                "channel_1": [1, 2, 3],
                "channel_2": [4, 5, 6],
                "channel_3": [7, 8, 9],
            }
        )

    def test_get_available_channel_names(
        self,
        dataframe_specified_channels,
        dataframe_node_based_channels,
        node_name,
        curr_channels,
        volt_channels,
    ):
        assert get_available_channel_names(
            dataframe_specified_channels.columns,
            current_channels=curr_channels,
            voltage_channels=[],
            nodes=node_name,
        ) == (["channel_3"], {SPECIFIED_CH_KEY: []}, {SPECIFIED_CH_KEY: ["channel_3"]})
        assert get_available_channel_names(
            dataframe_specified_channels.columns,
            current_channels=curr_channels,
            voltage_channels=volt_channels,
            nodes=node_name,
        ) == (
            ["channel_1", "channel_3"],
            {SPECIFIED_CH_KEY: ["channel_1"]},
            {SPECIFIED_CH_KEY: ["channel_3"]},
        )
        assert get_available_channel_names(
            dataframe_specified_channels.columns,
            current_channels=[],
            voltage_channels=volt_channels,
            nodes=[],
        ) == (["channel_1"], {SPECIFIED_CH_KEY: ["channel_1"]}, {SPECIFIED_CH_KEY: []})

        assert get_available_channel_names(
            dataframe_node_based_channels.columns,
            current_channels=[],
            voltage_channels=[],
            nodes=node_name,
        ) == (
            ["Vpt41", "Vpt42", "Ipt41", "Ipt42", "Ipt43", "Vpt32"],
            {"pt4": ["Vpt41", "Vpt42"], "pt3": ["Vpt32"]},
            {"pt4": ["Ipt41", "Ipt42", "Ipt43"], "pt3": []},
        )


class TestBreakerStatus:
    @pytest.fixture
    def ip_dataframe_brkr_status(self):
        return pd.DataFrame(
            {
                "other": [1, 2, 3, 4, 5],
                "xyz_brkr_ab": [0.7, 0.4, 0.3, 0.3, 0.3],
                "x_Solenoidabc": [0.8, 0.2, 0.8, 0.6, 0.1],
            }
        )

    @pytest.fixture
    def op_dataframe_brkr_status(self):
        return pd.DataFrame(
            {
                STATUS_COLUMN: [OFF_STATUS, ON_STATUS, ON_STATUS, ON_STATUS, ON_STATUS],
            }
        )

    def test_estimate_breaker_status(
        self, ip_dataframe_brkr_status, op_dataframe_brkr_status, breaker_keywords
    ):
        assert all(
            estimate_breaker_status(
                ip_dataframe_brkr_status,
                kwds_brkr=breaker_keywords,
            )
            == op_dataframe_brkr_status
        )


class TestDataAcquisition:
    @pytest.fixture
    def filepath_specified_channel_data(self):
        return os.path.join(os.getcwd(), "HiZLib", "tests", "123.csv")

    @pytest.fixture
    def op_data_acquisition_specified_channel_data(self):
        return pd.DataFrame(
            {
                "Time": [1, 2, 3, 4, 5],
                "channel_1": [1, 2, 3, 4, 5],
                "channel_3": [11, 12, 13, 14, 15],
                "Status": [OFF_STATUS, ON_STATUS, ON_STATUS, ON_STATUS, ON_STATUS],
            }
        )

    @pytest.fixture
    def filepath_node_based_data(self):
        return os.path.join(os.path.join(os.getcwd(), "HiZLib", "tests", "456.csv"))

    @pytest.fixture
    def op_data_acquisition_node_based_data(self):
        return pd.DataFrame(
            {
                "Time": [1, 2, 3, 4, 5],
                "Vpt41": [1, 2, 3, 4, 5],
                "Ipt42": [6, 7, 8, 9, 10],
                "Status": [OFF_STATUS, ON_STATUS, ON_STATUS, ON_STATUS, ON_STATUS],
            }
        )

    def test_data_acquisition(
        self,
        filepath_specified_channel_data,
        op_data_acquisition_specified_channel_data,
        filepath_node_based_data,
        op_data_acquisition_node_based_data,
        curr_channels,
        volt_channels,
        node_name,
        breaker_keywords,
    ):
        df_1, v_channels_1, i_channels_1 = data_acquisition(
            filepath_specified_channel_data,
            voltage_channels=volt_channels,
            current_channels=curr_channels,
            nodes=[],
        )
        assert (
            all(df_1 == op_data_acquisition_specified_channel_data)
            and (v_channels_1 == {"spec": ["channel_1"]})
            and (i_channels_1 == {"spec": ["channel_3"]})
        )

        df_2, v_channels_2, i_channels_2 = data_acquisition(
            filepath_node_based_data, voltage_channels=[], current_channels=[], nodes=node_name
        )
        assert (
            all(df_2 == op_data_acquisition_node_based_data)
            and (v_channels_2 == {"pt4": ["Vpt41"]})
            and (i_channels_2 == {"pt4": ["Ipt42"]})
        )
