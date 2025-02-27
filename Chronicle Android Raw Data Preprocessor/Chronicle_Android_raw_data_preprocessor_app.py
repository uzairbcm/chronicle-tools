from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sys
import traceback
import typing
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime as datetime_class
from datetime import tzinfo
from enum import Enum, StrEnum
from math import ceil
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def get_matching_files_from_folder(
    folder: Path | str,
    file_matching_pattern: str,
    ignore_names: list[str] | None = None,
) -> list[str]:
    """
    Retrieves a list of files from a specified folder that match a given pattern, excluding files with names in the ignore list.

    Args:
        folder (Path | str): The folder to search for files.
        file_matching_pattern (str): The regex pattern to match file names.
        ignore_names (list[str] | None): A list of names to ignore in the search. Defaults to None.

    Returns:
        list[str]: A list of matching file paths as strings.
    """
    LOGGER.debug(f"Getting matching files from folder: {folder} with pattern: {file_matching_pattern}")
    if not ignore_names:
        ignore_names = ["Preprocessed"]
    matching_files = [
        str(f)
        for f in Path(folder).rglob("**")
        if Path(f).is_file() and re.search(file_matching_pattern, str(f.name)) and all(ignored not in str(f) for ignored in ignore_names)
    ]
    LOGGER.debug(f"Found {len(matching_files)} matching files")
    return matching_files


def get_local_timezone() -> tzinfo | None:
    """
    Retrieves the local timezone of the system.

    Returns:
        tzinfo | None: The local timezone information.
    """
    # logger.debug("Getting local timezone")
    local_timezone = datetime_class.now(datetime.timezone.utc).astimezone().tzinfo
    # logger.debug(f"Local timezone determined: {local_timezone}")
    return local_timezone


class ChronicleDeviceType(StrEnum):
    """
    Enum representing different types of Chronicle devices.
    """

    AMAZON = "Amazon Fire"
    ANDROID = "Android"


class InteractionType(StrEnum):
    """
    Enum representing different types of interactions in the Chronicle data.
    """

    ACTIVITY_RESUMED = "Activity Resumed"
    ACTIVITY_PAUSED = "Activity Paused"
    APP_USAGE = "App Usage"
    END_OF_DAY = "End of Day"
    CONTINUE_PREVIOUS_DAY = "Continue Previous Day"
    CONFIGURATION_CHANGE = "Configuration Change"
    SYSTEM_INTERACTION = "System Interaction"
    USER_INTERACTION = "User Interaction"
    SHORTCUT_INVOCATION = "Shortcut Invocation"
    CHOOSER_ACTION = "Chooser Action"
    NOTIFICATION_SEEN = "Notification Seen"
    STANDBY_BUCKET_CHANGED = "Standby Bucket Changed"
    NOTIFICATION_INTERRUPTION = "Notification Interruption"
    SLICE_PINNED_PRIV = "Slice Pinned Priv"
    SLICE_PINNED_APP = "Slice Pinned App"
    SCREEN_INTERACTIVE = "Screen Interactive"
    SCREEN_NON_INTERACTIVE = "Screen Non-Interactive"
    KEYGUARD_SHOWN = "Keyguard Shown"
    KEYGUARD_HIDDEN = "Keyguard Hidden"
    FOREGROUND_SERVICE_START = "Foreground Service Start"
    FOREGROUND_SERVICE_STOP = "Foreground Service Stop"
    CONTINUING_FOREGROUND_SERVICE = "Continuing Foreground Service"
    ROLLOVER_FOREGROUND_SERVICE = "Rollover Foreground Service"
    ACTIVITY_STOPPED = "Activity Stopped"
    ACTIVITY_DESTROYED = "Activity Destroyed"
    FLUSH_TO_DISK = "Flush to Disk"
    DEVICE_SHUTDOWN = "Device Shutdown"
    DEVICE_STARTUP = "Device Startup"
    USER_UNLOCKED = "User Unlocked"
    USER_STOPPED = "User Stopped"
    LOCUS_ID_SET = "Locus ID Set"
    APP_COMPONENT_USED = "App Component Used"
    FILTERED_APP_RESUMED = "Filtered App Resumed"
    FILTERED_APP_PAUSED = "Filtered App Paused"
    FILTERED_APP_USAGE = "Filtered App Usage"
    END_OF_USAGE_MISSING = "End of Usage Missing"


class TimezoneHandlingOption(Enum):
    """
    Enum representing different options for handling timezones in the data.
    """

    REMOVE_DATA_WITH_NONPRIMARY_TIMEZONES = 0
    CONVERT_ALL_DATA_TO_PRIMARY_TIMEZONE = 1
    CONVERT_ALL_DATA_TO_LOCAL_TIMEZONE = 2
    CONVERT_ALL_DATA_TO_SPECIFIC_TIMEZONE = 3


@dataclass
class ChronicleAndroidRawDataPreprocessorOptions:
    """
    Options for preprocessing Chronicle Android raw data.

    Attributes:
        study_name (str): The name of the study.
        raw_data_folder (Path | str): Path to the folder containing raw data files.
        survey_data_folder (Path | str): Path to the folder containing survey data files.
        raw_data_file_pattern (str): Regex pattern to match raw data files.
        survey_data_file_pattern (str): Regex pattern to match survey data files.
        use_survey_data (bool): Flag indicating whether to use survey data.
        filter_file (Path | str): Path to the file containing filter information.
        apps_to_filter_dict (dict[str, str]): Dictionary of apps to filter.
        minimum_usage_duration (int): Minimum usage duration in seconds.
        custom_app_engagement_duration (int): Custom app engagement duration in seconds.
        timezone_handling_option (TimezoneHandlingOption): Option for handling timezones.
        specific_timezone (str | tzinfo | None): Specific timezone to use.
        correct_duplicate_event_timestamps (bool): Flag indicating whether to correct duplicate event timestamps.
        same_app_interaction_types_to_stop_usage_at (set[InteractionType]): Set of interaction types to stop usage at for the same app.
        other_interaction_types_to_stop_usage_at (set[InteractionType]): Set of other interaction types to stop usage at.
        interaction_types_to_remove (set[InteractionType]): Set of interaction types to remove.
    """

    study_name: str = ""
    raw_data_folder: Path | str = ""
    survey_data_folder: Path | str = ""
    raw_data_file_pattern: str = r"[\s\S]*.csv"
    survey_data_file_pattern: str = r"[\s\S]*(Survey)[\s\S]*.csv"
    use_survey_data: bool = False
    filter_file: Path | str = ""
    apps_to_filter_dict: dict[str, str] = field(default_factory=lambda: {"": ""})
    minimum_usage_duration: int = 1  # in seconds
    custom_app_engagement_duration: int = 300  # in seconds
    timezone_handling_option: TimezoneHandlingOption = TimezoneHandlingOption.REMOVE_DATA_WITH_NONPRIMARY_TIMEZONES
    specific_timezone: str | tzinfo | None = None
    correct_duplicate_event_timestamps: bool = True

    same_app_interaction_types_to_stop_usage_at: set[InteractionType] = field(default_factory=lambda: {InteractionType.ACTIVITY_PAUSED})

    other_interaction_types_to_stop_usage_at: set[InteractionType] = field(
        default_factory=lambda: {
            InteractionType.DEVICE_SHUTDOWN,
            InteractionType.ACTIVITY_RESUMED,
            InteractionType.FILTERED_APP_RESUMED,
            InteractionType.FILTERED_APP_USAGE,
        }
    )

    interaction_types_to_remove: set[InteractionType] = field(default_factory=lambda: {InteractionType.STANDBY_BUCKET_CHANGED})

    def __post_init__(self) -> None:
        """
        Post-initialization processing.
        """
        LOGGER.debug(f"Initialized ChronicleAndroidRawDataPreprocessorOptions with: {self}")

    @property
    def output_folder(self) -> Path:
        """
        Get the output folder path based on the raw data folder.

        Returns:
            Path: The output folder path.
        """
        output_folder = Path(self.raw_data_folder).parent
        LOGGER.debug(f"Output folder determined: {output_folder}")
        return output_folder


class ChronicleAndroidRawDataPreprocessor:
    """
    A class to preprocess Chronicle Android raw data.

    Attributes:
        options (ChronicleAndroidRawDataPreprocessorOptions): Options for the data preprocessing.
        current_participant_raw_data_df (pd.DataFrame): DataFrame containing the current participant's raw data.
        current_participant_id (str | None): The current participant's ID.
        participant_raw_data_df_target_child_only (pd.DataFrame): DataFrame containing only the target child's data.
        local_timezone (tzinfo): The local timezone.
        current_data_primary_timezone (tzinfo | None): The primary timezone of the current data.
    """

    def __init__(
        self,
        options: ChronicleAndroidRawDataPreprocessorOptions,
    ) -> None:
        """
        Initialize the ChronicleAndroidRawDataPreprocessor.

        Args:
            options (ChronicleAndroidRawDataPreprocessorOptions): Options for the data preprocessing.
        """
        LOGGER.debug(f"Initializing {self.__class__.__name__}")
        self.options = options

        if self.options.raw_data_folder is None or self.options.raw_data_folder == "":
            msg = "The raw data folder must be specified."
            LOGGER.error(msg)
            raise ValueError(msg)

        self.current_participant_raw_data_df: pd.DataFrame = pd.DataFrame([])
        self.current_participant_id: str | None = None
        self.participant_raw_data_df_target_child_only: pd.DataFrame = pd.DataFrame([])
        self.local_timezone = get_local_timezone()
        self.current_data_primary_timezone = None
        LOGGER.debug(f"{self.__class__.__name__} initialized successfully")

    @staticmethod
    def fix_timestamp_format(timestamp: str) -> str | None:
        """
        Fixes the format of the timestamp by adding milliseconds if missing.

        Args:
            timestamp (str): The timestamp string to be fixed.

        Returns:
            str | None: The fixed timestamp string or None if the format is incorrect.
        """
        if "Z" in timestamp:
            timestamp = timestamp.replace("Z", "+00:00")
        expected_timestamp_length = 25
        if len(timestamp) == expected_timestamp_length:  # Check if the timestamp is missing milliseconds based on length
            return timestamp[:-6] + ".000" + timestamp[-6:]  # Add .000 before the timezone info
        if len(timestamp) < expected_timestamp_length:
            LOGGER.error(f"Timestamp format is incorrect: {timestamp}")
            raise ValueError
        return timestamp

    def get_participant_id_from_data(self) -> str:
        """
        Gets the participant ID from the Chronicle raw data .csv file for a participant.

        Returns:
            str: The participant ID.
        """
        participant_id = str(self.current_participant_raw_data_df.iloc[1]["participant_id"])
        LOGGER.debug(f"Participant ID retrieved: {participant_id}")
        return participant_id

    def get_possible_device_model(self) -> ChronicleDeviceType:
        """
        Determines whether the Chronicle Android data is from an Amazon Fire tablet or a regular Android device
        based on the apps/services found within the data.

        Returns:
            ChronicleDeviceType: The type of device (AMAZON or ANDROID).
        """
        LOGGER.debug("Determining possible device model")
        amazon_apps = [
            "com.amazon.redstone",
            "com.amazon.firelauncher",
            "com.amazon.imp",
            "com.amazon.alta.h2clientservice",
            "com.amazon.media.session.monitor",
        ]
        if any(self.current_participant_raw_data_df["app_package_name"].str.contains("|".join(amazon_apps))):
            LOGGER.debug("Possible device model determined: Amazon Fire")
            return ChronicleDeviceType.AMAZON
        LOGGER.debug("Possible device model determined: Android")
        return ChronicleDeviceType.ANDROID

    def rename_interaction_types(self) -> None:
        """
        Renames interaction types in the dataframe based on the conversion dictionary.
        """
        LOGGER.debug("Renaming interaction types")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)
        self.current_participant_raw_data_df["interaction_type"] = self.current_participant_raw_data_df["interaction_type"].replace(
            ALL_INTERACTION_TYPES_MAP
        )
        LOGGER.debug("Interaction types renamed successfully")

    def remove_selected_interaction_types(self) -> None:
        """
        Removes selected interaction types from the dataframe.
        """
        LOGGER.debug("Removing selected interaction types")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df[
            ~self.current_participant_raw_data_df["interaction_type"].isin(self.options.interaction_types_to_remove)
        ]
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.sort_values("event_timestamp").reset_index(drop=True)
        LOGGER.debug("Selected interaction types removed successfully")

    def unalign_duplicate_event_timestamps(self) -> None:
        """
        Adjusts duplicate event timestamps by adding nanoseconds to ensure uniqueness.
        """
        LOGGER.debug("Unaligning duplicate event timestamps")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)
        duplicate_indices_groups_list = (
            self.current_participant_raw_data_df[self.current_participant_raw_data_df.duplicated(subset=["event_timestamp"], keep=False)]
            .groupby("event_timestamp")
            .apply(lambda x: list(x.index), include_groups=False)
            .reset_index(drop=True)
            .to_numpy()
            .tolist()
        )

        for group in duplicate_indices_groups_list:
            LOGGER.debug(
                f"{self.current_participant_raw_data_df['participant_id'].iloc[0]}: duplicates found for event_timestamp {self.current_participant_raw_data_df.loc[group[0], 'event_timestamp']}."
            )
            for i, idx in enumerate(group):
                self.current_participant_raw_data_df.loc[idx, "event_timestamp"] -= pd.Timedelta(i + 1, unit="nanoseconds")  # type: ignore

        self.current_participant_raw_data_df = self.current_participant_raw_data_df.sort_values("event_timestamp").reset_index(drop=True)
        LOGGER.debug("Duplicate event timestamps unaligned successfully")

    def apply_timezone_handling_options(self) -> None:
        """
        Applies the selected timezone handling options to the event timestamps.
        """

        def convert_to_timezone(timezone):
            LOGGER.info(f"Converting timestamps to timezone: {timezone}")
            self.current_participant_raw_data_df["event_timestamp"] = pd.to_datetime(
                self.current_participant_raw_data_df["event_timestamp"], utc=True
            ).dt.tz_convert(timezone)
            LOGGER.debug("Timezone conversion completed")

        # def is_same_timezone_with_dst(tz1, tz2):
        #     if tz1 is None or tz2 is None:
        #         return False
        #     return (
        #         tz1.utcoffset(None) == tz2.utcoffset(None)
        #         or tz1.utcoffset(None) == tz2.utcoffset(None) + pd.Timedelta(hours=1)
        #         or tz1.utcoffset(None) == tz2.utcoffset(None) - pd.Timedelta(hours=1)
        #     )

        LOGGER.info("Starting timezone handling operations...")
        initial_row_count = len(self.current_participant_raw_data_df)
        LOGGER.debug(f"Initial row count: {initial_row_count}")

        LOGGER.debug("Determining primary timezone...")
        timezones_series = pd.to_datetime(self.current_participant_raw_data_df["event_timestamp"], utc=False).apply(lambda x: x.tz)
        if not timezones_series.empty:
            self.current_data_primary_timezone = timezones_series.mode()[0]
            LOGGER.debug(f"Primary timezone determined: {self.current_data_primary_timezone}")
        else:
            LOGGER.warning("No timezone information found in data")

        timezone_option = self.options.timezone_handling_option
        LOGGER.info(f"Selected timezone handling option: {timezone_option}")

        if timezone_option == TimezoneHandlingOption.REMOVE_DATA_WITH_NONPRIMARY_TIMEZONES:
            LOGGER.info(f"Primary timezone: {self.current_data_primary_timezone}")
            if self.current_data_primary_timezone:
                timezones_series = pd.to_datetime(self.current_participant_raw_data_df["event_timestamp"], utc=False).apply(lambda x: x.tz)
                mask = (timezones_series == self.current_data_primary_timezone) & timezones_series.notna()
                self.current_participant_raw_data_df = self.current_participant_raw_data_df[mask]
                rows_removed = initial_row_count - len(self.current_participant_raw_data_df)
                LOGGER.warning(f"Removed {rows_removed} rows with non-primary timezones")

        elif timezone_option == TimezoneHandlingOption.CONVERT_ALL_DATA_TO_PRIMARY_TIMEZONE:
            LOGGER.info(f"Primary timezone: {self.current_data_primary_timezone}")
            if self.current_data_primary_timezone:
                convert_to_timezone(self.current_data_primary_timezone)
            else:
                LOGGER.warning("Could not determine primary timezone")

        elif timezone_option == TimezoneHandlingOption.CONVERT_ALL_DATA_TO_LOCAL_TIMEZONE:
            LOGGER.info(f"Converting to local timezone: {self.local_timezone}")
            convert_to_timezone(self.local_timezone)

        elif timezone_option == TimezoneHandlingOption.CONVERT_ALL_DATA_TO_SPECIFIC_TIMEZONE:
            if self.options.specific_timezone is None:
                LOGGER.error("No specific timezone provided")
                msg = "specific_timezone must be provided when using CONVERT_ALL_DATA_TO_SPECIFIC_TIMEZONE option"
                raise ValueError(msg)
            LOGGER.info(f"Converting to specific timezone: {self.options.specific_timezone}")
            convert_to_timezone(self.options.specific_timezone)

        else:
            LOGGER.error(f"Invalid timezone option: {timezone_option}")
            msg = f"Unsupported timezone handling option: {timezone_option}"
            raise ValueError(msg)

        self.current_participant_raw_data_df["event_timestamp"] = pd.to_datetime(self.current_participant_raw_data_df["event_timestamp"])

    def correct_event_timestamp_column(self) -> None:
        """
        Corrects the format of the event timestamp column and adjusts for timezone.
        """
        LOGGER.debug("Correcting event timestamp column")
        self.current_participant_raw_data_df["event_timestamp"] = self.current_participant_raw_data_df["event_timestamp"].apply(
            self.fix_timestamp_format
        )
        self.apply_timezone_handling_options()
        self.unalign_duplicate_event_timestamps()
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.sort_values("event_timestamp").reset_index(drop=True)
        LOGGER.debug("Event timestamp column corrected successfully")

    def correct_original_columns(self) -> None:
        """
        Corrects the original columns in the dataframe.
        """
        LOGGER.debug("Correcting original columns")
        self.current_participant_raw_data_df["username"] = self.current_participant_raw_data_df["username"].replace("Target child", "Target Child")
        self.rename_interaction_types()
        self.remove_selected_interaction_types()
        self.correct_event_timestamp_column()
        LOGGER.debug("Original columns corrected successfully")

    @typing.no_type_check
    def mark_data_time_gaps(self) -> None:
        """
        Marks gaps in the data by calculating the time difference between consecutive events.
        """
        LOGGER.debug("Marking data time gaps")
        self.current_participant_raw_data_df["data_time_gap_hours"] = 0

        for index, row in self.current_participant_raw_data_df.iterrows():
            if index == 0:
                continue

            backward_row = self.current_participant_raw_data_df.iloc[index - 1]
            time_since_last_data = (row["event_timestamp"] - backward_row["event_timestamp"]).total_seconds()
            time_since_last_data_hours = time_since_last_data / 3600
            time_since_last_data_hours_rounded = ceil(time_since_last_data_hours) if time_since_last_data_hours >= 0.5 else 0
            self.current_participant_raw_data_df.loc[index, "data_time_gap_hours"] = time_since_last_data_hours_rounded

        LOGGER.debug("Data time gaps marked successfully")

    def create_additional_columns(self) -> None:
        """
        Creates additional columns in the dataframe for date, day, weekday, hour, quarter, and possible device model.
        """
        LOGGER.debug("Creating additional columns")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        self.current_participant_raw_data_df["datetime_of_preprocessing"] = datetime_class.now(tz=get_local_timezone()).strftime("%m-%d-%Y %H:%M:%S")
        self.current_participant_raw_data_df["date"] = self.current_participant_raw_data_df["event_timestamp"].dt.date
        self.current_participant_raw_data_df["day"] = (self.current_participant_raw_data_df["event_timestamp"].dt.weekday + 1) % 7 + 1
        self.current_participant_raw_data_df["weekdayMF"] = (self.current_participant_raw_data_df["event_timestamp"].dt.weekday < 5).astype(int)
        self.current_participant_raw_data_df["weekdayMTh"] = (self.current_participant_raw_data_df["event_timestamp"].dt.weekday < 4).astype(int)
        self.current_participant_raw_data_df["weekdaySuTh"] = (
            (self.current_participant_raw_data_df["event_timestamp"].dt.weekday < 4)
            | (self.current_participant_raw_data_df["event_timestamp"].dt.weekday == 6)
        ).astype(int)
        self.current_participant_raw_data_df["hour"] = self.current_participant_raw_data_df["event_timestamp"].dt.hour
        self.current_participant_raw_data_df["quarter"] = self.current_participant_raw_data_df["event_timestamp"].dt.quarter
        self.current_participant_raw_data_df["possible_device_model"] = self.get_possible_device_model()

        self.mark_data_time_gaps()
        LOGGER.debug("Additional columns created successfully")

    def label_filtered_apps(self) -> None:
        """
        Filters out apps that are known to not be correctly accounted for by Chronicle, and apps that we have decided against counting as usage such as Settings.
        Currently filters based on the app package name and verifies the app package label.
        """
        LOGGER.debug("Labeling filtered apps")

        mask = self.current_participant_raw_data_df["app_package_name"].isin(self.options.apps_to_filter_dict.keys())

        for index, row in self.current_participant_raw_data_df[mask].iterrows():
            app_package_name = row["app_package_name"]
            app_label = row["application_label"]
            expected_labels = [label.strip() for label in self.options.apps_to_filter_dict[app_package_name].split(",")]

            if app_label not in expected_labels:
                LOGGER.warning(f"App label mismatch for package {app_package_name}: expected any of '{expected_labels}', found '{app_label}'")
                continue

            if row["interaction_type"] == InteractionType.ACTIVITY_RESUMED:
                self.current_participant_raw_data_df.loc[index, "interaction_type"] = InteractionType.FILTERED_APP_RESUMED  # type: ignore
            elif row["interaction_type"] in self.options.same_app_interaction_types_to_stop_usage_at:
                self.current_participant_raw_data_df.loc[index, "interaction_type"] = InteractionType.FILTERED_APP_PAUSED  # type: ignore

        LOGGER.debug("Filtered apps labeled successfully")

    @typing.no_type_check
    def process_filtered_app_usage_rows(self) -> None:
        """
        Processes raw data to determine start and stop
        timestamps for filtered app usage within a study period.
        """
        LOGGER.debug("Processing filtered app usage rows")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)
        self.current_participant_raw_data_df["start_timestamp"] = pd.NaT
        self.current_participant_raw_data_df["stop_timestamp"] = pd.NaT

        if (
            not self.current_participant_raw_data_df["interaction_type"]
            .isin([InteractionType.FILTERED_APP_RESUMED, InteractionType.FILTERED_APP_PAUSED])
            .any()
        ):
            msg = f"{self.current_participant_id} had no apparent usage for filtered out apps within the study period"
            LOGGER.warning(msg)
            print(msg)
            return self.current_participant_raw_data_df

        for index, row in self.current_participant_raw_data_df.iterrows():
            if row["interaction_type"] == InteractionType.FILTERED_APP_RESUMED:
                stop_timestamp_set = False
                for forward_index in range(index + 1, len(self.current_participant_raw_data_df)):
                    forward_row = self.current_participant_raw_data_df.iloc[forward_index]
                    if (
                        forward_row["app_package_name"] == row["app_package_name"]
                        and forward_row["interaction_type"] == InteractionType.FILTERED_APP_PAUSED
                    ) or forward_row["interaction_type"] in self.options.other_interaction_types_to_stop_usage_at:
                        with warnings.catch_warnings():
                            warnings.simplefilter(action="ignore", category=FutureWarning)
                            self.current_participant_raw_data_df.loc[index, "start_timestamp"] = row["event_timestamp"]
                            self.current_participant_raw_data_df.loc[index, "stop_timestamp"] = forward_row["event_timestamp"]
                        stop_timestamp_set = True
                        break

                if not stop_timestamp_set:
                    LOGGER.warning(
                        f"{self.current_participant_raw_data_df['participant_id'].iloc[0]}: Missing end timestamp for the final instance of app usage within the study period, using timestamp of final entry in data."
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                        self.current_participant_raw_data_df.loc[index, "start_timestamp"] = row["event_timestamp"]
                        self.current_participant_raw_data_df.loc[index, "interaction_type"] = InteractionType.END_OF_USAGE_MISSING

        self.current_participant_raw_data_df = self.current_participant_raw_data_df[
            ~(self.current_participant_raw_data_df["interaction_type"] == InteractionType.FILTERED_APP_PAUSED)
        ]

        self.current_participant_raw_data_df = self.current_participant_raw_data_df[
            ~(
                (self.current_participant_raw_data_df["interaction_type"] == InteractionType.FILTERED_APP_RESUMED)
                & (self.current_participant_raw_data_df["start_timestamp"].isna() | self.current_participant_raw_data_df["stop_timestamp"].isna())
            )
        ]

        self.current_participant_raw_data_df["interaction_type"] = self.current_participant_raw_data_df["interaction_type"].replace(
            InteractionType.FILTERED_APP_RESUMED, InteractionType.FILTERED_APP_USAGE
        )

        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)
        LOGGER.debug("Filtered app usage rows processed successfully")

    @typing.no_type_check
    def process_valid_app_usage_rows(self) -> None:
        """
        This function processes valid app usage data by adding columns for start and stop timestamps, date,
        and duration based on interaction types and event timestamps.

        Raises:
            pd.errors.EmptyDataError: If there is no valid app usage data during the study period.
        """
        LOGGER.debug("Processing valid app usage rows")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        self.current_participant_raw_data_df["start_timestamp"] = pd.NaT
        self.current_participant_raw_data_df["stop_timestamp"] = pd.NaT

        if self.current_participant_raw_data_df["interaction_type"].isin([InteractionType.ACTIVITY_RESUMED, InteractionType.ACTIVITY_PAUSED]).any():
            for index in range(len(self.current_participant_raw_data_df)):
                row_interaction_type = self.current_participant_raw_data_df.loc[index, "interaction_type"]
                row_app_package_name = self.current_participant_raw_data_df.loc[index, "app_package_name"]
                row_event_timestamp = self.current_participant_raw_data_df.loc[index, "event_timestamp"]
                stop_timestamp_set = False

                if row_interaction_type == InteractionType.ACTIVITY_RESUMED:
                    for forward_index in range(index + 1, len(self.current_participant_raw_data_df)):
                        forward_interaction_type = self.current_participant_raw_data_df.loc[forward_index, "interaction_type"]
                        forward_app_package_name = self.current_participant_raw_data_df.loc[forward_index, "app_package_name"]
                        forward_event_timestamp = self.current_participant_raw_data_df.loc[forward_index, "event_timestamp"]

                        if (
                            forward_app_package_name == row_app_package_name
                            and forward_interaction_type in self.options.same_app_interaction_types_to_stop_usage_at
                        ) or forward_interaction_type in self.options.other_interaction_types_to_stop_usage_at:
                            with warnings.catch_warnings():
                                warnings.simplefilter(action="ignore", category=FutureWarning)
                                self.current_participant_raw_data_df.loc[index, "start_timestamp"] = row_event_timestamp
                                self.current_participant_raw_data_df.loc[index, "stop_timestamp"] = forward_event_timestamp
                            stop_timestamp_set = True
                            break

                    if not stop_timestamp_set:
                        LOGGER.warning(
                            f"{self.current_participant_raw_data_df['participant_id'].iloc[0]}: Missing end timestamp for the final instance of app usage within the study period, using timestamp of final entry in data."
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter(action="ignore", category=FutureWarning)
                            self.current_participant_raw_data_df.loc[index, "start_timestamp"] = row_event_timestamp
                            self.current_participant_raw_data_df.loc[index, "interaction_type"] = InteractionType.END_OF_USAGE_MISSING

                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                        self.current_participant_raw_data_df.loc[index, "start_timestamp"] = row_event_timestamp
                        # self.current_participant_raw_data_df.loc[index, "stop_timestamp"] = forward_event_timestamp

            self.current_participant_raw_data_df = self.current_participant_raw_data_df[
                ~(self.current_participant_raw_data_df["interaction_type"] == InteractionType.ACTIVITY_PAUSED)
            ]

            self.current_participant_raw_data_df = self.current_participant_raw_data_df[
                ~(
                    (self.current_participant_raw_data_df["interaction_type"] == InteractionType.ACTIVITY_RESUMED)
                    & (self.current_participant_raw_data_df["start_timestamp"].isna() | self.current_participant_raw_data_df["stop_timestamp"].isna())
                )
            ]

            self.current_participant_raw_data_df["interaction_type"] = self.current_participant_raw_data_df["interaction_type"].replace(
                InteractionType.ACTIVITY_RESUMED, InteractionType.APP_USAGE
            )

        else:
            msg = f"{self.current_participant_id} had no apparent valid app usage within the study period"
            LOGGER.error(msg)
            raise pd.errors.EmptyDataError(msg)

        match self.options.timezone_handling_option:
            case TimezoneHandlingOption.REMOVE_DATA_WITH_NONPRIMARY_TIMEZONES:
                self.current_participant_raw_data_df["start_timestamp"] = pd.to_datetime(self.current_participant_raw_data_df["start_timestamp"])
                self.current_participant_raw_data_df["stop_timestamp"] = pd.to_datetime(self.current_participant_raw_data_df["stop_timestamp"])

            case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_PRIMARY_TIMEZONE:
                self.current_participant_raw_data_df["start_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["start_timestamp"], utc=True
                ).dt.tz_convert(self.current_data_primary_timezone)
                self.current_participant_raw_data_df["stop_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["stop_timestamp"], utc=True
                ).dt.tz_convert(self.current_data_primary_timezone)

            case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_LOCAL_TIMEZONE:
                self.current_participant_raw_data_df["start_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["start_timestamp"], utc=True
                ).dt.tz_convert(self.local_timezone)
                self.current_participant_raw_data_df["stop_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["stop_timestamp"], utc=True
                ).dt.tz_convert(self.local_timezone)

            case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_SPECIFIC_TIMEZONE:
                self.current_participant_raw_data_df["start_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["start_timestamp"], utc=True
                ).dt.tz_convert(self.options.specific_timezone)
                self.current_participant_raw_data_df["stop_timestamp"] = pd.to_datetime(
                    self.current_participant_raw_data_df["stop_timestamp"], utc=True
                ).dt.tz_convert(self.options.specific_timezone)

        self.fix_usages_split_across_dates()

        self.current_participant_raw_data_df.loc[
            self.current_participant_raw_data_df["interaction_type"] == InteractionType.APP_USAGE, "duration_seconds"
        ] = (self.current_participant_raw_data_df["stop_timestamp"] - self.current_participant_raw_data_df["start_timestamp"]).dt.total_seconds()

        self.current_participant_raw_data_df.loc[
            self.current_participant_raw_data_df["interaction_type"] == InteractionType.APP_USAGE, "duration_seconds"
        ] = self.current_participant_raw_data_df.loc[
            self.current_participant_raw_data_df["interaction_type"] == InteractionType.APP_USAGE, "duration_seconds"
        ].apply(lambda x: x if x >= self.options.minimum_usage_duration else None)

        self.current_participant_raw_data_df["duration_minutes"] = self.current_participant_raw_data_df["duration_seconds"] / 60

        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)
        LOGGER.debug("Valid app usage rows processed successfully")

    def convert_start_stop_timestamp_columns_to_simple_strings(self) -> None:
        """
        The function `convert_start_stop_timestamp_columns_to_simple_strings` converts the start and stop
        timestamp columns in a DataFrame to simple string format ("%H:%M:%S").
        """
        LOGGER.debug("Converting start and stop timestamp columns to simple strings")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        self.current_participant_raw_data_df["start_timestamp"] = self.current_participant_raw_data_df["start_timestamp"].dt.strftime("%H:%M:%S")
        self.current_participant_raw_data_df["stop_timestamp"] = self.current_participant_raw_data_df["stop_timestamp"].dt.strftime("%H:%M:%S")
        LOGGER.debug("Start and stop timestamp columns converted to simple strings successfully")

    @typing.no_type_check
    def add_app_usage_detail_columns(self) -> None:
        """
        The function `add_app_usage_detail_columns` adds new columns to a DataFrame based on app
        engagement and usage time gaps.
        """
        LOGGER.debug("Adding app usage detail columns")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        columns_defaults = {
            "valid_app_new_engage_30s": 0,
            f"valid_app_new_engage_custom_{self.options.custom_app_engagement_duration}s": 0,
            "valid_app_switched_app": 0,
            "valid_app_usage_time_gap_hours": 0.0,
            "any_app_new_engage_30s": 0,
            f"any_app_new_engage_custom_{self.options.custom_app_engagement_duration}s": 0,
            "any_app_switched_app": 0,
            "any_app_usage_time_gap_hours": 0.0,
            # "any_app_usage_flags": [],
        }

        def set_engagement_default_values(row, column, default_value):
            if "any_app" in column:
                return default_value if row["interaction_type"] in [InteractionType.APP_USAGE, InteractionType.FILTERED_APP_USAGE] else None
            else:
                return default_value if row["interaction_type"] == InteractionType.APP_USAGE else None

        for column, default_value in columns_defaults.items():
            self.current_participant_raw_data_df[column] = self.current_participant_raw_data_df.apply(
                lambda row: set_engagement_default_values(row, column, default_value), axis=1
            )

        app_usage_row_indices = self.current_participant_raw_data_df[
            self.current_participant_raw_data_df["interaction_type"] == InteractionType.APP_USAGE
        ].index

        interaction_types = self.current_participant_raw_data_df["interaction_type"].to_numpy()

        def traverse_app_usage_backward_rows(index, row):
            for backward_index in range(index - 1, -1, -1):
                backward_row = self.current_participant_raw_data_df.loc[backward_index]
                backward_interaction_type = interaction_types[backward_index]

                if backward_interaction_type != InteractionType.APP_USAGE:
                    continue

                if row["app_package_name"] != backward_row["app_package_name"]:
                    self.current_participant_raw_data_df.loc[index, "valid_app_switched_app"] = 1

                time_since_last_valid_app_use = (row["start_timestamp"] - backward_row["stop_timestamp"]).total_seconds()

                if time_since_last_valid_app_use > 30:
                    self.current_participant_raw_data_df.loc[index, "valid_app_new_engage_30s"] = 1

                if time_since_last_valid_app_use > self.options.custom_app_engagement_duration:
                    self.current_participant_raw_data_df.loc[index, f"valid_app_new_engage_custom_{self.options.custom_app_engagement_duration}s"] = 1

                self.current_participant_raw_data_df.loc[index, "valid_app_usage_time_gap_hours"] = time_since_last_valid_app_use // 3600

                break

        def traverse_backward_rows(index, row):
            for backward_index in range(index - 1, -1, -1):
                backward_row = self.current_participant_raw_data_df.loc[backward_index]
                backward_interaction_type = interaction_types[backward_index]

                if backward_interaction_type not in [InteractionType.APP_USAGE, InteractionType.FILTERED_APP_USAGE]:
                    continue

                if row["app_package_name"] != backward_row["app_package_name"]:
                    self.current_participant_raw_data_df.loc[index, "any_app_switched_app"] = 1

                time_since_last_any_app_use = (row["start_timestamp"] - backward_row["stop_timestamp"]).total_seconds()

                if time_since_last_any_app_use > 30:
                    self.current_participant_raw_data_df.loc[index, "any_app_new_engage_30s"] = 1

                if time_since_last_any_app_use > self.options.custom_app_engagement_duration:
                    self.current_participant_raw_data_df.loc[index, f"any_app_new_engage_custom_{self.options.custom_app_engagement_duration}s"] = 1

                self.current_participant_raw_data_df.loc[index, "any_app_usage_time_gap_hours"] = time_since_last_any_app_use // 3600

                break

            if row["interaction_type"] == InteractionType.APP_USAGE:
                traverse_app_usage_backward_rows(index, row)

        def set_first_app_use_engagement_values(index, custom_gap, is_app_usage):
            if not is_app_usage:
                self.current_participant_raw_data_df.loc[index, f"any_app_new_engage_custom_{custom_gap}s"] = 1
                self.current_participant_raw_data_df.loc[index, "any_app_new_engage_30s"] = 1
            if is_app_usage:
                self.current_participant_raw_data_df.loc[index, f"valid_app_new_engage_custom_{custom_gap}s"] = 1
                self.current_participant_raw_data_df.loc[index, "valid_app_new_engage_30s"] = 1

        def process_row_app_usage_details(index, row, first_app_set):
            if not first_app_set:
                if row["interaction_type"] in [InteractionType.APP_USAGE, InteractionType.FILTERED_APP_USAGE]:
                    set_first_app_use_engagement_values(
                        index, self.options.custom_app_engagement_duration, row["interaction_type"] == InteractionType.APP_USAGE
                    )
                    first_app_set = True
            elif index > 0 and row["interaction_type"] in [InteractionType.APP_USAGE, InteractionType.FILTERED_APP_USAGE]:
                if index == app_usage_row_indices[0]:
                    if row["interaction_type"] == InteractionType.APP_USAGE:
                        set_first_app_use_engagement_values(index, self.options.custom_app_engagement_duration, True)
                else:
                    traverse_backward_rows(index, row)
            return first_app_set

        first_app_set = False
        for index, row in self.current_participant_raw_data_df.iterrows():
            first_app_set = process_row_app_usage_details(index, row, first_app_set)
        LOGGER.debug("App usage detail columns added successfully")

    def fix_usages_split_across_dates(self) -> None:
        """
        Fixes the calculation of usage durations when an instance of usage is split across two different dates
        """
        LOGGER.debug("Fixing usages split across dates")
        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        usages_split_across_dates = self.current_participant_raw_data_df[
            (self.current_participant_raw_data_df["start_timestamp"].dt.date != self.current_participant_raw_data_df["stop_timestamp"].dt.date)
            & (self.current_participant_raw_data_df["interaction_type"] == InteractionType.APP_USAGE)
        ]

        if not usages_split_across_dates.empty:
            LOGGER.warning(f"Usage split across dates found: {usages_split_across_dates}")

        for idx in usages_split_across_dates.index:
            row = self.current_participant_raw_data_df.loc[idx]
            start_ts = row["start_timestamp"]
            stop_ts = row["stop_timestamp"]
            event_ts = row["event_timestamp"]
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=UserWarning)
                match self.options.timezone_handling_option:
                    case TimezoneHandlingOption.REMOVE_DATA_WITH_NONPRIMARY_TIMEZONES:
                        stop_ts1 = pd.to_datetime(f"{start_ts.date()!s} 23:59:59.999998{str(event_ts)[-6:]}")
                        start_ts2 = pd.to_datetime(f"{stop_ts.date()!s} 0:00:00.000001{str(event_ts)[-6:]}")
                        event_ts2 = pd.to_datetime(f"{stop_ts.date()!s}T00:00:00.001{str(event_ts)[-6:]}")

                    case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_PRIMARY_TIMEZONE:
                        stop_ts1 = pd.to_datetime(f"{start_ts.date()!s} 23:59:59.999998{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.current_data_primary_timezone
                        )
                        start_ts2 = pd.to_datetime(f"{stop_ts.date()!s} 0:00:00.000001{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.current_data_primary_timezone
                        )
                        event_ts2 = pd.to_datetime(f"{stop_ts.date()!s}T00:00:00.001{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.current_data_primary_timezone
                        )

                    case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_LOCAL_TIMEZONE:
                        stop_ts1 = pd.to_datetime(f"{start_ts.date()!s} 23:59:59.999998{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.local_timezone
                        )
                        start_ts2 = pd.to_datetime(f"{stop_ts.date()!s} 0:00:00.000001{str(event_ts)[-6:]}", utc=True).tz_convert(self.local_timezone)
                        event_ts2 = pd.to_datetime(f"{stop_ts.date()!s}T00:00:00.001{str(event_ts)[-6:]}", utc=True).tz_convert(self.local_timezone)

                    case TimezoneHandlingOption.CONVERT_ALL_DATA_TO_SPECIFIC_TIMEZONE:
                        stop_ts1 = pd.to_datetime(f"{start_ts.date()!s} 23:59:59.999998{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.options.specific_timezone
                        )
                        start_ts2 = pd.to_datetime(f"{stop_ts.date()!s} 0:00:00.000001{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.options.specific_timezone
                        )
                        event_ts2 = pd.to_datetime(f"{stop_ts.date()!s}T00:00:00.001{str(event_ts)[-6:]}", utc=True).tz_convert(
                            self.options.specific_timezone
                        )

            self.current_participant_raw_data_df.loc[idx, "stop_timestamp"] = stop_ts1
            row2 = row.copy()
            row2["start_timestamp"] = start_ts2
            row2["event_timestamp"] = event_ts2
            self.current_participant_raw_data_df.loc[:, len(self.current_participant_raw_data_df.index)] = row2
        LOGGER.debug("Usages split across dates fixed successfully")

    def create_target_child_only_df(self) -> None:
        """
        Filters the current participant raw data DataFrame to include only rows where the 'username'
        column contains the string 'Target Child'. The filtered DataFrame is stored in the
        'participant_raw_data_df_target_child_only' attribute.

        Returns:
            None
        """
        LOGGER.debug("Creating target child only DataFrame")
        self.participant_raw_data_df_target_child_only = self.current_participant_raw_data_df[
            self.current_participant_raw_data_df["username"].astype(str).str.contains("Target Child")
        ]
        LOGGER.debug("Target child only DataFrame created successfully")

    def check_data_for_disordered_timestamps(self) -> None:
        """
        Checks the Chronicle raw data dataframe for occurrences where the start timestamp of app usage is later than the stop timestamp.

        Raises:
            ValueError: If disordered timestamps are detected.
        """
        LOGGER.debug("Checking data for disordered timestamps")
        disordered_timestamps = self.current_participant_raw_data_df[
            self.current_participant_raw_data_df["start_timestamp"] > self.current_participant_raw_data_df["stop_timestamp"]
        ]

        if len(disordered_timestamps.index) > 0:
            LOGGER.error(f"Found {len(disordered_timestamps.index)} disordered timestamps")
            print(disordered_timestamps[["start_timestamp", "stop_timestamp"]])
            msg = f"There were {len(disordered_timestamps.index)} occurrences of the start timestamp being later than the stop timestamp, which should be impossible."
            raise ValueError(msg)
        LOGGER.debug("No disordered timestamps found")

    @typing.no_type_check
    def mark_app_usage_flags(self) -> None:
        def get_time_gap_flag(time_gap: float, thresholds: list[float]):
            """Return the appropriate time gap flag based on the time gap in hours and custom thresholds."""
            thresholds.sort(reverse=True)
            for threshold in thresholds:
                if time_gap >= threshold:
                    return f">{threshold}-HR TIME GAP"
            return None

        def get_duration_flag(duration_minutes: float, thresholds: list[float]):
            """Return the appropriate duration flag based on the duration in minutes and custom thresholds."""
            duration_hours = duration_minutes / 60
            thresholds.sort(reverse=True)
            for threshold in thresholds:
                if duration_hours >= threshold:
                    return f">{threshold}-HR APP USAGE"
            return None

        def add_app_usage_flags(row: pd.Series, time_gap_thresholds: list[float], duration_thresholds: list[float]):
            """Return a list of flags for a given row based on time gap and duration."""
            time_gap_flag = get_time_gap_flag(row["data_time_gap_hours"], time_gap_thresholds)
            duration_flag = get_duration_flag(row["duration_minutes"], duration_thresholds)
            return [flag for flag in [time_gap_flag, duration_flag] if flag]

        # Apply the add_app_usage_flags function to each row in the DataFrame
        time_gap_thresholds = duration_thresholds = [3, 6, 12, 24]
        self.current_participant_raw_data_df["app_usage_flags"] = self.current_participant_raw_data_df.apply(
            lambda row: add_app_usage_flags(row, time_gap_thresholds, duration_thresholds), axis=1
        )

    def finalize_and_save_preprocessed_data_df(self, raw_data_filename: str) -> Path:
        """
        Makes final edits and saves the final preprocessed data within a subfolder of the main folder.

        Args:
            raw_data_filename (str): The name of the raw data file being processed.

        Returns:
            Path: The path to the folder where the preprocessed data is saved.
        """
        LOGGER.debug(f"Finalizing and saving preprocessed data for {raw_data_filename}")
        preprocessed_data_save_folder = Path(f"{self.options.output_folder}/Chronicle Android Preprocessed Data")

        preprocessed_data_save_folder.mkdir(parents=True, exist_ok=True)

        save_name = f"{preprocessed_data_save_folder}/{raw_data_filename.replace('Raw', 'Preprocessed')}"

        self.current_participant_raw_data_df = self.current_participant_raw_data_df.reset_index(drop=True)

        if self.current_participant_raw_data_df.empty:
            # If the dataframe is empty, save an empty dataframe
            LOGGER.warning("Dataframe is empty, saving empty dataframe")
            self.current_participant_raw_data_df.to_csv(save_name, index=False)
            return preprocessed_data_save_folder
        else:
            # Select specific columns to save
            self.current_participant_raw_data_df = self.current_participant_raw_data_df[
                [
                    "study_id",
                    "participant_id",
                    "possible_device_model",
                    "interaction_type",
                    "date",
                    "start_timestamp",
                    "stop_timestamp",
                    "duration_seconds",
                    "duration_minutes",
                    "username",
                    "application_label",
                    "app_package_name",
                    "event_timestamp",
                    "timezone",
                    "day",
                    "weekdayMF",
                    "weekdayMTh",
                    "weekdaySuTh",
                    "data_time_gap_hours",
                    "valid_app_new_engage_30s",
                    f"valid_app_new_engage_custom_{self.options.custom_app_engagement_duration}s",
                    "valid_app_switched_app",
                    "valid_app_usage_time_gap_hours",
                    "any_app_new_engage_30s",
                    f"any_app_new_engage_custom_{self.options.custom_app_engagement_duration}s",
                    "any_app_switched_app",
                    "any_app_usage_time_gap_hours",
                    "app_usage_flags",
                    "datetime_of_preprocessing",
                ]
            ]

        self.check_data_for_disordered_timestamps()

        self.convert_start_stop_timestamp_columns_to_simple_strings()

        self.remove_selected_interaction_types()

        self.current_participant_raw_data_df.to_csv(save_name, index=False)
        LOGGER.debug(f"Preprocessed data saved to {save_name}")

        return preprocessed_data_save_folder

    def preprocess_Chronicle_Android_raw_data_file_without_survey_data(self, raw_data_file: Path | str) -> Path:
        """
        Preprocesses a Chronicle Android raw data file without using survey data.
        Filters specific apps listed in the 'filter_apps_from_df' function.

        Args:
            raw_data_file (Path | str): Path to the Chronicle Android raw data file.

        Returns:
            Path: The path to the folder where the preprocessed data is saved.

        Raises:
            pd.errors.EmptyDataError: If there is no valid app usage data during the study period.

        Notes:
            - The function reads the raw data file and processes it to correct and add columns.
            - It filters and labels app usage data based on predefined criteria.
            - If no valid app usage data is found, it handles the exception and logs a message.
            - The processed data is saved to the specified output folder.
        """
        LOGGER.debug(f"Preprocessing Chronicle Android raw data file {raw_data_file} without survey data")
        self.current_participant_raw_data_df = pd.read_csv(raw_data_file)
        self.current_participant_id = self.get_participant_id_from_data()
        self.correct_original_columns()
        self.create_additional_columns()
        self.label_filtered_apps()
        self.process_filtered_app_usage_rows()

        try:
            self.process_valid_app_usage_rows()
        except pd.errors.EmptyDataError:
            # Handle the case where no valid app usage data is found
            msg = f"{self.current_participant_id}: No valid app usage during the study period."
            LOGGER.warning(msg)
            print(msg)

            # Drop rows with missing usernames and save the preprocessed data
            self.current_participant_raw_data_df = self.current_participant_raw_data_df.dropna(subset="username")
            preprocessed_data_save_folder = self.finalize_and_save_preprocessed_data_df(raw_data_filename=Path(raw_data_file).name)
        else:
            self.add_app_usage_detail_columns()
            self.mark_app_usage_flags()
            # Save the preprocessed data
            preprocessed_data_save_folder = self.finalize_and_save_preprocessed_data_df(raw_data_filename=Path(raw_data_file).name)

        LOGGER.debug(f"Preprocessed data for {raw_data_file} saved to {preprocessed_data_save_folder}")
        return preprocessed_data_save_folder

    def preprocess_Chronicle_Android_raw_data_folder(self) -> Path | None:
        """
        Preprocesses an entire folder of Chronicle Android raw data files and saves the results in the output folder.

        Returns:
            Path | None: The path to the folder where the preprocessed data is saved, or None if no files are found.

        Notes:
            - The function scans the specified folder for raw data files, ignoring folders named "Do Not Use", "Archive", and "Processed".
            - If no raw data files are found, it prints a message and exits.
            - Each raw data file is processed individually. If a file is empty, it is skipped with a prompt to the user.
        """
        LOGGER.debug("Preprocessing Chronicle Android raw data folder")
        ignore_names = ["Do Not Use", "Archive", "Processed"]

        Chronicle_Android_raw_data_files = get_matching_files_from_folder(
            folder=self.options.raw_data_folder,
            file_matching_pattern=self.options.raw_data_file_pattern,
            ignore_names=ignore_names,
        )

        try:
            if not len(Chronicle_Android_raw_data_files) > 0:
                # If no raw data files are found, log a warning and print a message
                LOGGER.error("No Chronicle Android raw data files found in the provided folder")
                # return print("No Chronicle Android raw data files found in the provided folder.")
            for raw_data_file in Chronicle_Android_raw_data_files:
                preprocessed_data_save_folder = self.preprocess_Chronicle_Android_raw_data_file_without_survey_data(raw_data_file=raw_data_file)
        except pd.errors.EmptyDataError as e:
            # Handle the case where a raw data file is empty
            msg = f"The Chronicle Android raw data file {raw_data_file} was empty."
            LOGGER.error(msg)
            raise pd.errors.EmptyDataError(msg) from None

        # Update the options dictionary with additional information
        options_dict_output = self.options.__dict__.copy()
        options_dict_output.update(
            {
                "raw_data_files": Chronicle_Android_raw_data_files,
                "date_and_time": datetime_class.now(tz=get_local_timezone()).strftime("%m-%d-%Y %H:%M:%S"),
                "preprocessed_data_save_folder": preprocessed_data_save_folder,
            }
        )

        options_df_output = pd.DataFrame([options_dict_output]).T
        options_df_output.to_csv(
            preprocessed_data_save_folder  # type: ignore
            / "Chronicle_Android_raw_data_preprocessor_app_options.csv",  # type: ignore
            index=True,
        )

        LOGGER.debug(f"Preprocessed data folder: {preprocessed_data_save_folder}")
        return preprocessed_data_save_folder


class QDialogABCMeta(type(QDialog), ABC.__class__):
    """
    QDialogABCMeta is a metaclass that combines the functionality of QDialog and ABC (Abstract Base Class).
    This allows for the creation of abstract base classes that are also QDialog subclasses.
    """

    pass


class BaseTableWindow(QDialog, ABC, metaclass=QDialogABCMeta):
    """
    BaseTableWindow is an abstract base class for creating a dialog window with a table.
    It provides methods to initialize the UI, center the dialog on the parent window,
    set up the table with data, and handle checkbox interactions.

    Attributes:
        table (QTableWidget): The table widget used to display data.
        selected_interaction_types (set): A set of selected interaction types.
    """

    def __init__(self, parent_: ChronicleAndroidRawDataPreprocessorGUI) -> None:
        """
        Initialize the BaseTableWindow.

        Args:
            parent_ (ChronicleAndroidRawDataPreprocessorApp): The parent application instance.
        """
        LOGGER.debug(f"Initializing {self.__class__.__name__}")
        super().__init__(parent_)
        self.setMaximumSize(800, 600)
        self._init_UI()
        self._center_on_parent()
        LOGGER.debug(f"{self.__class__.__name__} initialized successfully")

    @abstractmethod
    def _init_UI(self) -> None:
        """
        Initialize the user interface. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _center_on_parent(self) -> None:
        """
        Center the dialog on the parent window.
        """
        LOGGER.debug(f"Centering {self.__class__.__name__} on parent")
        parent_geometry = self.parent().geometry()
        parent_center = parent_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(parent_center)
        self.move(self_geometry.topLeft())
        LOGGER.debug(f"{self.__class__.__name__} centered on parent")

    def setup_table(self, column_count: int, headers: list, data: dict, checkbox_column: bool = False) -> None:
        """
        Set up the table with the given data.

        Args:
            column_count (int): The number of columns in the table.
            headers (list): The headers for the table columns.
            data (dict): The data to populate the table, with keys as original values and values as converted values.
            checkbox_column (bool): Whether to include a checkbox column. Defaults to False.
        """
        LOGGER.debug(f"Setting up table in {self.__class__.__name__}")
        self.table = QTableWidget()
        self.table.setColumnCount(column_count)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        for i, (original, converted) in enumerate(data.items()):
            self.table.insertRow(i)
            original_item = QTableWidgetItem(original)
            converted_item = QTableWidgetItem(converted.value)
            original_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            converted_item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)  # type: ignore
            converted_item.setFlags(converted_item.flags() & ~Qt.ItemIsEditable)  # type: ignore
            self.table.setItem(i, 0, original_item)
            self.table.setItem(i, 1, converted_item)

            if checkbox_column:
                checkbox = QCheckBox()
                if converted in self.selected_interaction_types:
                    checkbox.setChecked(True)
                checkbox_widget = QWidget()
                checkbox_layout = QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(Qt.AlignCenter)  # type: ignore
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                checkbox.stateChanged.connect(lambda state, conv=converted: self._update_selected_values(state, conv))
                self.table.setCellWidget(i, 2, checkbox_widget)
                if converted in self.get_checked_and_disabled_interaction_types():
                    checkbox.setDisabled(True)
                    checkbox.setChecked(True)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # type: ignore
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # type: ignore
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #dcdcdc;
                font-size: 12pt;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #dcdcdc;
                font-size: 12pt;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)
        LOGGER.debug(f"Table setup completed in {self.__class__.__name__}")

    @abstractmethod
    def get_checked_and_disabled_interaction_types(self) -> set:
        """
        Get the set of checked and disabled values. This method must be implemented by subclasses.

        Returns:
            set: A set of checked and disabled interaction types.
        """
        raise NotImplementedError

    def _update_selected_values(self, checkbox_state: int, interaction_type: InteractionType) -> None:
        """
        Update the selected interaction types based on the checkbox state.

        Args:
            checkbox_state (int): The state of the checkbox (checked or unchecked).
            interaction_type (InteractionType): The interaction type associated with the checkbox.
        """
        LOGGER.debug(f"Updating selected interaction types in {self.__class__.__name__}")
        if checkbox_state == Qt.Checked:  # type: ignore
            self.selected_interaction_types.add(interaction_type)
        else:
            self.selected_interaction_types.remove(interaction_type)
        self.update_parent_options()
        LOGGER.debug(f"Selected interaction types updated in {self.__class__.__name__}")

    def _resize_to_fit_table(self) -> None:
        """
        Resize the window to fit the table's contents.
        """
        LOGGER.debug(f"Resizing {self.__class__.__name__} to fit table")
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        table_width = (
            self.table.verticalHeader().width()  # type: ignore
            + self.table.horizontalHeader().length()  # type: ignore
            + self.table.frameWidth() * 2
        )
        table_height = self.table.verticalHeader().length() + self.table.frameWidth() * 2  # type: ignore
        for row in range(self.table.rowCount()):
            table_height += self.table.rowHeight(row)
        self.resize(min(table_width, 800), min(table_height, 600))
        LOGGER.debug(f"{self.__class__.__name__} resized to fit table")

    @abstractmethod
    def update_parent_options(self) -> None:
        """
        Update the parent options. This method must be implemented by subclasses.
        """
        raise NotImplementedError


class AllInteractionTypeWindow(BaseTableWindow):
    """
    AllInteractionTypeWindow is a dialog window that displays all interaction types.
    It inherits from BaseTableWindow and initializes the UI with a table showing
    original and converted interaction types.
    """

    def __init__(self, parent_: ChronicleAndroidRawDataPreprocessorGUI) -> None:
        """
        Initialize the AllInteractionTypeWindow.

        Args:
            parent_ (ChronicleAndroidRawDataPreprocessorApp): The parent application instance.
        """
        self.all_interaction_types_map = ALL_INTERACTION_TYPES_MAP
        super().__init__(parent_)

    @typing.override
    def _init_UI(self) -> None:
        """
        Initialize the user interface for AllInteractionTypeWindow.
        """
        LOGGER.debug(f"Initializing UI in {self.__class__.__name__}")
        self.setWindowTitle("All Interaction Types")
        self.setGeometry(100, 100, 800, 600)
        self.setup_table(2, ["Original Interaction Type", "Converted Interaction Type"], self.all_interaction_types_map)
        self._resize_to_fit_table()
        LOGGER.debug(f"UI initialized in {self.__class__.__name__}")

    @typing.override
    def update_parent_options(self) -> None:
        """
        Update the parent options. This method is required by the abstract base class but is not used in this class.
        """
        pass

    @typing.override
    def get_checked_and_disabled_interaction_types(self) -> set:
        """
        Get the set of interaction types that are checked and disabled.

        Returns:
            set: A set of checked and disabled interaction types.
        """
        return set()


class RemoveInteractionTypeWindow(BaseTableWindow):
    """
    RemoveInteractionTypeWindow is a dialog window that allows the user to select interaction types to remove.
    It inherits from BaseTableWindow and initializes the UI with a table showing
    original and converted interaction types with checkboxes for removal.
    """

    def __init__(self, parent_: ChronicleAndroidRawDataPreprocessorGUI) -> None:
        """
        Initialize the RemoveInteractionTypeWindow.

        Args:
            parent_ (ChronicleAndroidRawDataPreprocessorApp): The parent application instance.
        """
        self.selected_interaction_types = parent_.preprocessor_options.interaction_types_to_remove or set()
        self.possible_interaction_types_to_remove = POSSIBLE_INTERACTION_TYPES_TO_REMOVE
        super().__init__(parent_)

    @typing.override
    def _init_UI(self) -> None:
        """
        Initialize the user interface for RemoveInteractionTypeWindow.
        """
        LOGGER.debug(f"Initializing UI in {self.__class__.__name__}")
        self.setWindowTitle("Interaction Types to Remove")
        self.setup_table(
            3,
            ["Original Interaction Type", "Converted Interaction Type", "Remove from Final Output?"],
            self.possible_interaction_types_to_remove,
            checkbox_column=True,
        )
        self._resize_to_fit_table()
        LOGGER.debug(f"UI initialized in {self.__class__.__name__}")

    @typing.override
    def update_parent_options(self) -> None:
        """
        Update the parent options with the selected interaction types to remove.
        """
        LOGGER.debug(f"Updating parent options in {self.__class__.__name__}")
        self.parent().preprocessor_options.interaction_types_to_remove = self.selected_interaction_types
        LOGGER.debug(f"Parent options updated in {self.__class__.__name__}")

    @typing.override
    def get_checked_and_disabled_interaction_types(self) -> set:
        """
        Get the set of interaction types that are checked and disabled.

        Returns:
            set: A set of checked and disabled interaction types.
        """
        return set()


class StopSameAppInteractionTypeWindow(BaseTableWindow):
    """
    StopSameAppInteractionTypeWindow is a dialog window that allows the user to select interaction types
    to stop usage at within the same app. It inherits from BaseTableWindow and initializes the UI with a table
    showing original and converted interaction types with checkboxes for stopping usage.
    """

    def __init__(self, parent_: ChronicleAndroidRawDataPreprocessorGUI) -> None:
        """
        Initialize the StopSameAppInteractionTypeWindow.

        Args:
            parent_ (ChronicleAndroidRawDataPreprocessorApp): The parent application instance.
        """
        self.selected_interaction_types = parent_.preprocessor_options.same_app_interaction_types_to_stop_usage_at or set()
        self.possible_same_app_interaction_types_to_stop_usage_at = POSSIBLE_SAME_APP_INTERACTION_TYPES_TO_STOP_USAGE_AT
        super().__init__(parent_)

    @typing.override
    def _init_UI(self) -> None:
        """
        Initialize the user interface for StopSameAppInteractionTypeWindow.
        """
        LOGGER.debug(f"Initializing UI in {self.__class__.__name__}")
        self.setWindowTitle("Same App Interaction Types to Stop Usage At")
        self.setup_table(
            3,
            ["Original Interaction Type", "Converted Interaction Type", "Stop Usage at Interaction Type?"],
            self.possible_same_app_interaction_types_to_stop_usage_at,
            checkbox_column=True,
        )
        self._resize_to_fit_table()
        LOGGER.debug(f"UI initialized in {self.__class__.__name__}")

    @typing.override
    def get_checked_and_disabled_interaction_types(self) -> set:
        """
        Get the set of interaction types that are checked and disabled.

        Returns:
            set: A set of checked and disabled interaction types.
        """
        return {
            InteractionType.ACTIVITY_PAUSED,
        }

    @typing.override
    def update_parent_options(self) -> None:
        """
        Update the parent options with the selected interaction types to stop usage at within the same app.
        """
        LOGGER.debug(f"Updating parent options in {self.__class__.__name__}")
        self.parent().preprocessor_options.same_app_interaction_types_to_stop_usage_at = self.selected_interaction_types
        LOGGER.debug(f"Parent options updated in {self.__class__.__name__}")


class StopOtherInteractionTypeWindow(BaseTableWindow):
    """
    StopOtherInteractionTypeWindow is a dialog window that allows the user to select interaction types
    to stop usage at for other apps. It inherits from BaseTableWindow and initializes the UI with a table
    showing original and converted interaction types with checkboxes for stopping usage.
    """

    def __init__(self, parent_: ChronicleAndroidRawDataPreprocessorGUI) -> None:
        """
        Initialize the StopOtherInteractionTypeWindow.

        Args:
            parent_ (ChronicleAndroidRawDataPreprocessorApp): The parent application instance.
        """
        self.selected_interaction_types = parent_.preprocessor_options.other_interaction_types_to_stop_usage_at or set()
        self.possible_other_interaction_types_to_stop_usage_at = POSSIBLE_OTHER_INTERACTION_TYPES_TO_STOP_USAGE_AT
        super().__init__(parent_)

    @typing.override
    def _init_UI(self) -> None:
        """
        Initialize the user interface for StopOtherInteractionTypeWindow.
        """
        LOGGER.debug(f"Initializing UI in {self.__class__.__name__}")
        self.setWindowTitle("Other Interaction Types to Stop Usage At")
        self.setup_table(
            3,
            ["Original Interaction Type", "Converted Interaction Type", "Stop Usage at Interaction Type?"],
            self.possible_other_interaction_types_to_stop_usage_at,
            checkbox_column=True,
        )
        self._resize_to_fit_table()
        LOGGER.debug(f"UI initialized in {self.__class__.__name__}")

    @typing.override
    def get_checked_and_disabled_interaction_types(self) -> set:
        """
        Get the set of interaction types that are checked and disabled.

        Returns:
            set: A set of checked and disabled interaction types.
        """
        return {
            InteractionType.ACTIVITY_RESUMED,
            InteractionType.FILTERED_APP_RESUMED,
            InteractionType.FILTERED_APP_USAGE,
        }

    @typing.override
    def update_parent_options(self) -> None:
        """
        Update the parent options with the selected interaction types to stop usage at for other apps.
        """
        LOGGER.debug(f"Updating parent options in {self.__class__.__name__}")
        self.parent().preprocessor_options.other_interaction_types_to_stop_usage_at = self.selected_interaction_types
        LOGGER.debug(f"Parent options updated in {self.__class__.__name__}")


class PreprocessorThreadWorker(QThread):
    """
    A worker thread for running the data preprocessing in the background.

    Attributes:
        finished (pyqtSignal): Signal emitted when preprocessing is finished.
        error (pyqtSignal): Signal emitted when an error occurs.
        preprocessor_options (ChronicleAndroidRawDataPreprocessorOptions): Options for the data preprocessing.
        continue_processing (bool): Flag indicating whether to continue processing.
    """

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, preprocessor_options: ChronicleAndroidRawDataPreprocessorOptions) -> None:
        """
        Initialize the PreprocessorThreadWorker.

        Args:
            preprocessor_options (ChronicleAndroidRawDataPreprocessorOptions): Options for the data preprocessing.
        """
        LOGGER.debug(f"Initializing {self.__class__.__name__}")
        super().__init__()
        self.preprocessor_options = preprocessor_options
        self.continue_processing = True
        LOGGER.debug(f"{self.__class__.__name__} initialized successfully")

    def run(self) -> None:
        """
        Run the data preprocessing in the background.
        """
        LOGGER.debug(f"Running {self.__class__.__name__}")
        try:
            # Check if the raw data folder is selected
            if not self.preprocessor_options.raw_data_folder:
                self.error.emit("Please select the Chronicle Android raw data folder.")
                LOGGER.error("No raw data folder selected.")
                return

            # Initialize the preprocessor with the provided options
            self.preprocessor = ChronicleAndroidRawDataPreprocessor(options=self.preprocessor_options)
            # Run the preprocessing and get the save folder path
            preprocessed_data_save_folder = self.preprocessor.preprocess_Chronicle_Android_raw_data_folder()
            # Emit the finished signal
            self.finished.emit()
            LOGGER.debug("Preprocessing completed successfully.")

        except pd.errors.EmptyDataError as e:
            # Handle empty data error
            self.error.emit(f"{e!s} The script has been stopped.")
            LOGGER.exception(f"EmptyDataError: {e!s}")

        except Exception as e:
            # Handle any other exceptions
            self.error.emit(traceback.format_exc())
            LOGGER.exception(f"Exception: {traceback.format_exc()}")


class ChronicleAndroidRawDataPreprocessorGUI(QWidget):
    """
    A GUI application for preprocessing Chronicle Android raw data.

    This class provides a graphical user interface (GUI) for users to select and configure
    various preprocessing options for Chronicle Android raw data. It allows users to select
    raw data folders, filter files, interaction types to remove or stop usage at, and other
    preprocessing options. The class also handles loading and saving configuration settings,
    and running the preprocessing in a separate thread.
    """

    def __init__(self) -> None:
        """
        Initialize the ChronicleAndroidRawDataPreprocessorApp.

        This method sets up the initial state of the application, initializes the UI,
        loads the configuration, and updates the labels.
        """
        super().__init__()
        LOGGER.debug(f"Initializing {self.__class__.__name__}")
        self.preprocessor_options = ChronicleAndroidRawDataPreprocessorOptions()
        self.local_timezone = get_local_timezone()
        self._init_UI()
        self._load_and_set_config()
        self._update_labels()
        LOGGER.debug(f"{self.__class__.__name__} initialized successfully")

    def _update_interaction_types_to_remove_label(self) -> None:
        """
        Update the label to show the selected interaction types to remove.

        This method updates the label that displays the interaction types that are selected
        to be removed from the final output.
        """
        LOGGER.debug("Updating interaction types to remove label")
        if self.preprocessor_options.interaction_types_to_remove:
            self.selected_interaction_types_to_remove_label.setText(
                f"Interaction Types Being Removed:\n\n {', '.join(self.preprocessor_options.interaction_types_to_remove)}"
            )
        else:
            self.selected_interaction_types_to_remove_label.setText("Interaction Types Being Removed:\n\n None")

        self.adjustSize()  # Adjust the size of the window
        LOGGER.debug("Interaction types to remove label updated")

    def _update_interaction_types_to_stop_usage_at_labels(self) -> None:
        """
        Update the labels to show the selected interaction types to stop usage at.

        This method updates the labels that display the interaction types that are selected
        to stop usage at for both same app and other app interaction types.
        """
        LOGGER.debug("Updating interaction types to stop usage at labels")
        if self.preprocessor_options.same_app_interaction_types_to_stop_usage_at:
            self.selected_same_app_interaction_types_to_stop_usage_at_label.setText(
                f"Same App Interaction Types to Stop Usage At:\n\n {', '.join(self.preprocessor_options.same_app_interaction_types_to_stop_usage_at)}"
            )
        else:
            self.selected_same_app_interaction_types_to_stop_usage_at_label.setText("Same App Interaction Types to Stop Usage At:\n\n None")

        if self.preprocessor_options.other_interaction_types_to_stop_usage_at:
            self.selected_other_app_interaction_types_to_stop_usage_at_label.setText(
                f"Other Interaction Types to Stop Usage At:\n\n {', '.join(self.preprocessor_options.other_interaction_types_to_stop_usage_at)}"
            )
        else:
            self.selected_other_app_interaction_types_to_stop_usage_at_label.setText("Other Interaction Types to Stop Usage At:\n\n None")

        self.adjustSize()  # Adjust the size of the window
        LOGGER.debug("Interaction types to stop usage at labels updated")

    def _update_labels(self) -> None:
        """
        Update all labels in the UI.

        This method updates all the labels in the UI to reflect the current state of the
        preprocessor options.
        """
        LOGGER.debug("Updating all labels")
        self._update_interaction_types_to_remove_label()
        self._update_interaction_types_to_stop_usage_at_labels()
        self.adjustSize()  # Adjust the size of the window
        LOGGER.debug("All labels updated")

    def _display_interaction_type_window(self, mode: str) -> None:
        """
        Display the interaction type window based on the given mode.

        Args:
            mode (str): The mode for the interaction type window. Can be one of "all", "stop_at_same",
                        "stop_at_other", or "remove".
        """
        LOGGER.debug(f"Displaying interaction type window for mode: {mode}")
        match mode:
            case "all":
                interaction_type_conversion_window = AllInteractionTypeWindow(self)
            case "stop_at_same":
                interaction_type_conversion_window = StopSameAppInteractionTypeWindow(self)
            case "stop_at_other":
                interaction_type_conversion_window = StopOtherInteractionTypeWindow(self)
            case "remove":
                interaction_type_conversion_window = RemoveInteractionTypeWindow(self)
            case _:
                interaction_type_conversion_window = AllInteractionTypeWindow(self)
        interaction_type_conversion_window.exec_()
        self._update_labels()
        LOGGER.debug(f"Interaction type window for mode: {mode} displayed and labels updated")

    def _open_output_folder(self) -> None:
        """
        Open the output folder in the file explorer.

        This method opens the output folder specified in the preprocessor options using the
        default file explorer.
        """
        if self.preprocessor_options:
            LOGGER.debug("Opening output folder")
            os.startfile(self.preprocessor_options.output_folder)
            LOGGER.debug("Output folder opened")

    @staticmethod
    def is_json_serializable(obj) -> bool:
        """
        Check if an object is JSON serializable.

        Args:
            obj: The object to check.

        Returns:
            bool: True if the object is JSON serializable, False otherwise.
        """
        try:
            json.dumps(obj)
        except (TypeError, OverflowError):
            return False
        else:
            return True

    @staticmethod
    def convert_to_boolean(value) -> bool:
        """
        Convert a value to a boolean.

        Args:
            value: The value to convert.

        Returns:
            bool: The converted boolean value.

        Raises:
            ValueError: If the value cannot be converted to a boolean.
        """
        true_values = {
            "true",
            "1",
            "yes",
            "y",
            "t",
            "on",
        }
        false_values = {
            "false",
            "0",
            "no",
            "n",
            "f",
            "off",
        }

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in true_values:
                return True
            elif value_lower in false_values:
                return False

        msg = f"Cannot convert {value} to boolean"
        raise ValueError(msg)

    def _select_and_validate_raw_data_folder(self) -> None:
        """
        Select and validate the raw data folder.

        This method opens a file dialog for the user to select the raw data folder and validates
        the selected folder.
        """
        LOGGER.debug("Selecting raw data folder")
        current_raw_data_folder_label = self.raw_data_folder_label.text().strip()
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Raw Data Folder")

        if selected_folder and Path(selected_folder).is_dir():
            # Update the preprocessor options and label with the selected folder
            self.preprocessor_options.raw_data_folder = selected_folder
            self.raw_data_folder_label.setText(selected_folder)
            LOGGER.debug(f"Selected raw data folder: {selected_folder}")
        else:
            # Revert to the previous label if an invalid folder is selected
            self.raw_data_folder_label.setText(current_raw_data_folder_label)
            LOGGER.debug("Invalid folder selected or no folder selected, reset to previous value")

        self.adjustSize()  # Adjust the size of the window

    def _process_filter_file(self, filter_file_df: pd.DataFrame) -> None:
        """
        Process the filter file.

        Args:
            filter_file_df (pd.DataFrame): The DataFrame containing the filter file data.

        Raises:
            ValueError: If the filter file does not contain the expected columns.
        """
        LOGGER.debug("Processing filter file")
        expected_columns = ["app_package_name", "known_application_labels", "app_filter_category", "filter_bool"]

        if list(filter_file_df.columns) != expected_columns:
            msg = "The file with the apps to filter did not contain the expected columns: 'app_package_name', 'known_application_labels', 'app_filter_category', 'filter_bool'."
            raise ValueError(msg)

        # Create a dictionary of apps to filter based on the filter file
        apps_to_filter_dict = {
            row["app_package_name"]: row["known_application_labels"]
            for _, row in filter_file_df[filter_file_df["filter_bool"].apply(self.convert_to_boolean)].iterrows()
        }

        # Update the preprocessor options with the apps to filter
        self.preprocessor_options.apps_to_filter_dict = apps_to_filter_dict

    # def _display_app_filter_selection_window(self) -> None:
    # """
    # Display the app filter selection window.

    # This method opens a window for the user to select the apps to filter.
    # """
    # filter_categories_window = FilterCategoriesWindow(self, filter_categories_dict=self.filter_categories_dict, selected_apps={})
    # filter_categories_window.exec_()
    # LOGGER.debug(f"Filter categories window displayed")

    def _select_validate_process_filter_file(self) -> None:
        """
        Select, validate, and process the filter file.

        This method opens a file dialog for the user to select the filter file, validates the selected
        file, and processes it.
        """
        LOGGER.debug("Selecting file with apps to filter")
        current_filter_file_label = self.filter_file_label.text().strip()
        selected_file, _ = QFileDialog.getOpenFileName(self, "Select Apps to Filter File", filter="Excel Files (*.xlsx);;CSV Files (*.csv)")

        if selected_file:
            try:
                # Read the selected file based on its extension
                if selected_file.endswith(".xlsx"):
                    filter_file_df = pd.read_excel(selected_file)
                elif selected_file.endswith(".csv"):
                    filter_file_df = pd.read_csv(selected_file)
                else:
                    msg = "Unsupported file format"
                    raise ValueError(msg)

                # Update the label with the selected file path
                self.filter_file_label.setText(selected_file)
                LOGGER.debug(f"Selected apps to filter file: {selected_file}")

                # Process the filter file
                self._process_filter_file(filter_file_df=filter_file_df)
            except Exception as e:
                # Revert to the previous label if an error occurs
                self.filter_file_label.setText(current_filter_file_label)
                LOGGER.exception(f"Error processing file: {e}")
                QMessageBox.warning(self, "Error", f"Failed to process the selected file: {e}")
            else:
                # Update the preprocessor options with the selected file
                self.preprocessor_options.filter_file = selected_file
        else:
            # Revert to the previous label if no file is selected
            self.filter_file_label.setText(current_filter_file_label)
            LOGGER.debug("No file selected, reset to previous value")

        self.adjustSize()  # Adjust the size of the window

    def _select_and_validate_survey_data_folder(self) -> None:
        """
        Select and validate the survey data folder.

        This method opens a file dialog for the user to select the survey data folder and validates
        the selected folder.
        """
        LOGGER.debug("Selecting survey data folder")
        current_survey_data_folder_label = self.survey_data_folder_label.text().strip()
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Survey Data Folder")
        if selected_folder and Path(selected_folder).is_dir():
            # Update the preprocessor options and label with the selected folder
            self.preprocessor_options.survey_data_folder = selected_folder
            self.survey_data_folder_label.setText(selected_folder)
            LOGGER.debug(f"Selected survey data folder: {selected_folder}")
        else:
            # Revert to the previous label if an invalid folder is selected
            self.survey_data_folder_label.setText(current_survey_data_folder_label)
            LOGGER.debug("Invalid folder selected or no folder selected, reset to previous value")

        self.adjustSize()  # Adjust the size of the window

    def _survey_data_checkbox_update(self) -> None:
        """
        Update the visibility of the survey data folder group based on the checkbox state.

        This method updates the visibility of the survey data folder group based on whether the
        "Use Survey Data" checkbox is checked.
        """
        # Show or hide the survey data folder group based on the checkbox state
        if self.use_survey_data_checkbox.isChecked():
            self.survey_data_folder_group.setVisible(True)
        else:
            self.survey_data_folder_group.setVisible(False)
        self.preprocessor_options.use_survey_data = self.use_survey_data_checkbox.isChecked()

        self.adjustSize()  # Adjust the size of the window

    def _validate_text_is_int(self, text_widget: QLineEdit, default_value: str) -> None:
        """
        Validate that the text in the given widget is an integer.

        Args:
            text_widget (QLineEdit): The text widget to validate.
            default_value (str): The default value to set if the text is not a valid integer.
        """
        text = text_widget.text().strip()

        if not text:
            # Set to default value if the text is empty
            text_widget.setText(default_value)
            LOGGER.debug(f"{text_widget} was empty, reset to default value {default_value}")
            return

        try:
            # Try to convert the text to an integer
            int(text)
        except ValueError:
            # Set to default value if the text is not a valid integer
            text_widget.setText(default_value)
            LOGGER.debug(f"{text_widget} contained non-integer value, reset to default value {default_value}")

    def _center_window(self):
        """
        Center the window on the screen.

        This method centers the main window on the screen.
        """
        LOGGER.debug("Centering window")
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())  # Get the screen number where the cursor is located
        centerPoint = QApplication.desktop().screenGeometry(screen).center()  # Get the center point of the screen
        frameGm.moveCenter(centerPoint)  # Move the frame to the center point
        self.move(frameGm.topLeft())  # Move the window to the top-left corner of the frame

    def _init_UI(self) -> None:
        """
        Initialize the user interface.

        This method sets up the user interface, including creating and arranging all the widgets
        and layouts.
        """
        LOGGER.debug("Initializing UI")
        self.setWindowTitle("Chronicle Android Raw Data Preprocessor")

        main_layout = QVBoxLayout()

        # Create horizontal layout for the top section
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._create_raw_data_folder_group())
        top_layout.addSpacing(10)
        top_layout.addWidget(self._create_filter_group())
        main_layout.addLayout(top_layout)
        main_layout.addSpacing(10)

        # Create horizontal layout for the middle section
        middle_layout = QHBoxLayout()
        middle_layout.addWidget(self._create_interaction_types_to_stop_usage_at_group())
        middle_layout.addSpacing(10)
        middle_layout.addWidget(self._create_interaction_types_to_remove_group())
        main_layout.addLayout(middle_layout)
        main_layout.addSpacing(10)

        # Create horizontal layout for the bottom section
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._create_minimum_usage_duration_group())
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self._create_custom_app_engagement_duration_group())
        main_layout.addLayout(bottom_layout)
        main_layout.addSpacing(10)

        # Add additional timezone handling options group
        main_layout.addWidget(self._create_timezone_handling_options_group())
        main_layout.addSpacing(10)

        # Add additional options group
        main_layout.addWidget(self._create_additional_options_group())
        main_layout.addSpacing(10)

        # Add survey data folder group
        self.survey_data_folder_group = self._create_survey_data_folder_group()
        main_layout.addWidget(self.survey_data_folder_group)
        main_layout.addSpacing(10)

        # Add bottom buttons group
        main_layout.addLayout(self._create_bottom_buttons())

        self.setLayout(main_layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._update_labels()
        self._center_window()
        self._update_labels()
        self.adjustSize()

    def _create_raw_data_folder_group(self) -> QGroupBox:
        """
        Create the raw data folder selection group.

        Returns:
            QGroupBox: The group box containing the raw data folder selection widgets.
        """
        LOGGER.debug("Creating raw data folder selection group")
        raw_data_folder_group = QGroupBox("Raw Data Folder Selection")
        raw_data_folder_layout = QVBoxLayout()

        # Create and configure the button for selecting the raw data folder
        self.select_raw_data_folder_button = QPushButton("Select Raw Data Folder")
        self.select_raw_data_folder_button.clicked.connect(self._select_and_validate_raw_data_folder)
        self.select_raw_data_folder_button.setStyleSheet("QPushButton { padding: 10px; }")

        button_layout = QHBoxLayout()
        self.select_raw_data_folder_button = QPushButton("Select Raw Data Folder")
        self.select_raw_data_folder_button.clicked.connect(self._select_and_validate_raw_data_folder)
        self.select_raw_data_folder_button.setStyleSheet("QPushButton { padding: 10px; }")

        button_layout.addStretch(1)
        button_layout.addWidget(self.select_raw_data_folder_button)
        button_layout.addStretch(1)

        raw_data_folder_layout.addLayout(button_layout)
        raw_data_folder_layout.addSpacing(10)

        # Create and configure the label for the raw data folder
        label_layout = QHBoxLayout()
        self.raw_data_folder_label = QLabel("Select the folder containing the Chronicle Android raw data")
        self.raw_data_folder_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.raw_data_folder_label.setWordWrap(True)
        self.raw_data_folder_label.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }")

        label_layout.addStretch(1)
        label_layout.addWidget(self.raw_data_folder_label)
        label_layout.addStretch(1)

        raw_data_folder_layout.addLayout(label_layout)
        raw_data_folder_layout.addSpacing(10)

        raw_data_folder_group.setLayout(raw_data_folder_layout)
        return raw_data_folder_group

    def _create_filter_group(self) -> QGroupBox:
        """
        Create the filter group.

        Returns:
            QGroupBox: The group box containing the filter selection widgets.
        """
        LOGGER.debug("Creating filter group")
        filter_group = QGroupBox("Apps to Filter")
        filter_layout = QVBoxLayout()

        # Create and configure the button for selecting the filter file
        button_layout = QHBoxLayout()
        self.select_filter_file_button = QPushButton("Select File Containing Apps to Filter")
        self.select_filter_file_button.clicked.connect(self._select_validate_process_filter_file)
        self.select_filter_file_button.setStyleSheet("QPushButton { padding: 10px; }")
        self.select_filter_file_button.setFixedWidth(200)  # Set fixed width

        button_layout.addStretch()
        button_layout.addWidget(self.select_filter_file_button)
        button_layout.addStretch()

        filter_layout.addLayout(button_layout)
        filter_layout.addSpacing(10)

        # Create and configure the label for the filter file
        label_layout = QHBoxLayout()
        self.filter_file_label = QLabel("Select the file containing the apps to filter")
        self.filter_file_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.filter_file_label.setWordWrap(True)
        self.filter_file_label.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }")

        label_layout.addStretch()
        label_layout.addWidget(self.filter_file_label)
        label_layout.addStretch()

        filter_layout.addLayout(label_layout)
        filter_layout.addSpacing(10)

        filter_group.setLayout(filter_layout)
        return filter_group

    def _create_interaction_types_to_stop_usage_at_group(self) -> QGroupBox:
        """
        Create the interaction types to stop usage at group.

        This method creates a group box containing widgets for selecting interaction types
        to stop usage at for both same app and other app interaction types.

        Returns:
            QGroupBox: The group box containing the interaction types to stop usage at widgets.
        """
        LOGGER.debug("Creating interaction types to stop usage at group")
        interaction_types_to_stop_usage_at_group = QGroupBox("Interaction Types to Stop Usage At")
        interaction_types_to_stop_usage_at_layout = QHBoxLayout()

        # Same App Interaction Types
        same_app_interaction_types_to_stop_usage_at_layout = QVBoxLayout()

        # Create and configure the button for selecting same app interaction types
        same_app_button_layout = QHBoxLayout()
        self.select_same_app_interaction_types_to_stop_usage_at_button = QPushButton("Select Same App Interaction Types to Stop Usage At")
        self.select_same_app_interaction_types_to_stop_usage_at_button.clicked.connect(
            lambda: self._display_interaction_type_window(mode="stop_at_same")
        )
        self.select_same_app_interaction_types_to_stop_usage_at_button.setStyleSheet("QPushButton { padding: 10px; }")
        self.select_same_app_interaction_types_to_stop_usage_at_button.setFixedWidth(300)  # Set fixed width

        same_app_button_layout.addStretch()
        same_app_button_layout.addWidget(self.select_same_app_interaction_types_to_stop_usage_at_button)
        same_app_button_layout.addStretch()

        same_app_interaction_types_to_stop_usage_at_layout.addLayout(same_app_button_layout)

        # Create and configure the label for same app interaction types
        same_app_label_layout = QHBoxLayout()
        self.selected_same_app_interaction_types_to_stop_usage_at_label = QLabel("Same App Interaction Types to Stop Usage At:\n\n None")
        self.selected_same_app_interaction_types_to_stop_usage_at_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.selected_same_app_interaction_types_to_stop_usage_at_label.setWordWrap(True)
        self.selected_same_app_interaction_types_to_stop_usage_at_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }"
        )

        same_app_label_layout.addStretch()
        same_app_label_layout.addWidget(self.selected_same_app_interaction_types_to_stop_usage_at_label)
        same_app_label_layout.addStretch()

        same_app_interaction_types_to_stop_usage_at_layout.addLayout(same_app_label_layout)

        # Other Interaction Types
        other_app_interaction_types_to_stop_usage_at_layout = QVBoxLayout()

        # Create and configure the button for selecting other app interaction types
        other_app_button_layout = QHBoxLayout()
        self.select_other_app_interaction_types_to_stop_usage_at_button = QPushButton("Select Other Interaction Types to Stop Usage At")
        self.select_other_app_interaction_types_to_stop_usage_at_button.clicked.connect(
            lambda: self._display_interaction_type_window(mode="stop_at_other")
        )
        self.select_other_app_interaction_types_to_stop_usage_at_button.setStyleSheet("QPushButton { padding: 10px; }")
        self.select_other_app_interaction_types_to_stop_usage_at_button.setFixedWidth(300)  # Set fixed width

        other_app_button_layout.addStretch()
        other_app_button_layout.addWidget(self.select_other_app_interaction_types_to_stop_usage_at_button)
        other_app_button_layout.addStretch()

        other_app_interaction_types_to_stop_usage_at_layout.addLayout(other_app_button_layout)

        # Create and configure the label for other app interaction types
        other_app_label_layout = QHBoxLayout()
        self.selected_other_app_interaction_types_to_stop_usage_at_label = QLabel("Other Interaction Types to Stop Usage At:\n\n None")
        self.selected_other_app_interaction_types_to_stop_usage_at_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.selected_other_app_interaction_types_to_stop_usage_at_label.setWordWrap(True)
        self.selected_other_app_interaction_types_to_stop_usage_at_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }"
        )

        other_app_label_layout.addStretch()
        other_app_label_layout.addWidget(self.selected_other_app_interaction_types_to_stop_usage_at_label)
        other_app_label_layout.addStretch()

        other_app_interaction_types_to_stop_usage_at_layout.addLayout(other_app_label_layout)

        # Add both layouts to the main interaction layout
        interaction_types_to_stop_usage_at_layout.addLayout(same_app_interaction_types_to_stop_usage_at_layout)
        interaction_types_to_stop_usage_at_layout.addLayout(other_app_interaction_types_to_stop_usage_at_layout)
        interaction_types_to_stop_usage_at_layout.addSpacing(10)

        interaction_types_to_stop_usage_at_group.setLayout(interaction_types_to_stop_usage_at_layout)
        return interaction_types_to_stop_usage_at_group

    def _create_interaction_types_to_remove_group(self) -> QGroupBox:
        """
        Create the interaction types to remove group.

        This method creates a group box containing widgets for selecting interaction types
        to remove from the final output.

        Returns:
            QGroupBox: The group box containing the interaction types to remove widgets.
        """
        LOGGER.debug("Creating interaction types to remove group")
        interaction_types_to_remove_group = QGroupBox("Interaction Types to Remove from Final Output")
        interaction_types_to_remove_layout = QVBoxLayout()

        # Create and configure the button for selecting interaction types to remove
        button_layout = QHBoxLayout()
        self.select_interaction_types_to_remove_button = QPushButton("Select Interaction Types to Remove from Final Output")
        self.select_interaction_types_to_remove_button.clicked.connect(lambda: self._display_interaction_type_window(mode="remove"))
        self.select_interaction_types_to_remove_button.setStyleSheet("QPushButton { padding: 10px; }")

        button_layout.addStretch()
        button_layout.addWidget(self.select_interaction_types_to_remove_button)
        button_layout.addStretch()

        interaction_types_to_remove_layout.addLayout(button_layout)

        # Create and configure the label for interaction types to remove
        label_layout = QHBoxLayout()
        self.selected_interaction_types_to_remove_label = QLabel("Interaction Types Being Removed:\n\n None")
        self.selected_interaction_types_to_remove_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.selected_interaction_types_to_remove_label.setWordWrap(True)
        self.selected_interaction_types_to_remove_label.setStyleSheet(
            "QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }"
        )
        self.selected_interaction_types_to_remove_label.setFixedWidth(300)

        label_layout.addStretch()
        label_layout.addWidget(self.selected_interaction_types_to_remove_label)
        label_layout.addStretch()

        interaction_types_to_remove_layout.addLayout(label_layout)

        interaction_types_to_remove_group.setLayout(interaction_types_to_remove_layout)
        return interaction_types_to_remove_group

    def _create_minimum_usage_duration_group(self) -> QGroupBox:
        """
        Create the minimum usage duration group.

        This method creates a group box containing widgets for setting the minimum usage duration
        required for an instance of app use to be counted.

        Returns:
            QGroupBox: The group box containing the minimum usage duration widgets.
        """
        LOGGER.debug("Creating minimum usage duration group")
        minimum_usage_duration_group = QGroupBox("Minimum Usage Duration")
        minimum_usage_duration_layout = QFormLayout()

        # Create and configure the label for minimum usage duration
        self.minimum_usage_duration_label = QLabel("Enter the minimum amount of usage in seconds required for an instance of app use to be counted:")
        self.minimum_usage_duration_label.setWordWrap(True)
        self.minimum_usage_duration_label.setAlignment(Qt.AlignJustify)  # type: ignore

        # Create and configure the entry for minimum usage duration
        self.minimum_usage_duration_entry = QLineEdit()
        minimum_usage_duration_default_value = "0"
        self.minimum_usage_duration_entry.setText(minimum_usage_duration_default_value)
        self.minimum_usage_duration_entry.editingFinished.connect(
            lambda: self._validate_text_is_int(text_widget=self.minimum_usage_duration_entry, default_value=minimum_usage_duration_default_value)
        )
        self.minimum_usage_duration_entry.setFixedWidth(30)
        self.minimum_usage_duration_entry.setAlignment(Qt.AlignCenter)  # type: ignore
        self.minimum_usage_duration_entry.setStyleSheet("QLineEdit { padding: 2px; border: 1px solid #bdc3c7; border-radius: 4px; }")

        # Create horizontal layout for label
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        label_layout.addWidget(self.minimum_usage_duration_label)
        label_layout.addStretch()

        # Create horizontal layout for entry
        entry_layout = QHBoxLayout()
        entry_layout.addStretch()
        entry_layout.addWidget(self.minimum_usage_duration_entry)
        entry_layout.addStretch()

        # Add layouts to main layout
        minimum_usage_duration_layout.addRow(label_layout)
        minimum_usage_duration_layout.addRow(entry_layout)
        minimum_usage_duration_group.setLayout(minimum_usage_duration_layout)
        return minimum_usage_duration_group

    def _create_custom_app_engagement_duration_group(self) -> QGroupBox:
        """
        Create the custom app engagement duration group.

        This method creates a group box containing widgets for setting a custom duration of usage
        for app engagement.

        Returns:
            QGroupBox: The group box containing the custom app engagement duration widgets.
        """
        LOGGER.debug("Creating custom app engagement duration group")
        custom_app_engagement_duration_group = QGroupBox("Custom App Engagement Duration")
        custom_app_engagement_duration_layout = QFormLayout()

        # Create and configure the label for custom app engagement duration
        self.custom_app_engagement_duration_label = QLabel(
            "Enter a custom duration of usage in seconds for app engagement (default of 30 seconds is already included in output):"
        )
        self.custom_app_engagement_duration_label.setWordWrap(True)
        self.custom_app_engagement_duration_label.setAlignment(Qt.AlignJustify)  # type: ignore

        # Create and configure the entry for custom app engagement duration
        self.custom_app_engagement_duration_entry = QLineEdit()
        custom_app_engagement_duration_default_value = "300"
        self.custom_app_engagement_duration_entry.setText(custom_app_engagement_duration_default_value)
        self.custom_app_engagement_duration_entry.editingFinished.connect(
            lambda: self._validate_text_is_int(
                text_widget=self.custom_app_engagement_duration_entry, default_value=custom_app_engagement_duration_default_value
            )
        )
        self.custom_app_engagement_duration_entry.setFixedWidth(40)
        self.custom_app_engagement_duration_entry.setAlignment(Qt.AlignCenter)  # type: ignore
        self.custom_app_engagement_duration_entry.setStyleSheet("QLineEdit { padding: 2px; border: 1px solid #bdc3c7; border-radius: 4px; }")

        # Create horizontal layout for label
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        label_layout.addWidget(self.custom_app_engagement_duration_label)
        label_layout.addStretch()

        # Create horizontal layout for entry
        entry_layout = QHBoxLayout()
        entry_layout.addStretch()
        entry_layout.addWidget(self.custom_app_engagement_duration_entry)
        entry_layout.addStretch()

        # Add layouts to main layout
        custom_app_engagement_duration_layout.addRow(label_layout)
        custom_app_engagement_duration_layout.addRow(entry_layout)
        custom_app_engagement_duration_group.setLayout(custom_app_engagement_duration_layout)
        return custom_app_engagement_duration_group

    def _create_timezone_handling_options_group(self) -> QGroupBox:
        """
        Create the timezone handling options group.

        This method creates a group box containing widgets for selecting timezone handling options.

        Returns:
            QGroupBox: The group box containing the timezone handling options widgets.
        """
        LOGGER.debug("Creating timezone handling options group")
        timezone_handling_group = QGroupBox("Timezone Handling Options")
        timezone_handling_layout = QHBoxLayout()

        # Create and configure the radio buttons for timezone handling options
        self.radio_remove_nonprimary = QRadioButton("Remove Data with Non-Primary Timezones")
        self.radio_convert_to_primary = QRadioButton("Convert All Data to Primary Timezone")
        self.radio_convert_to_local = QRadioButton("Convert All Data to Local Timezone")
        self.radio_convert_to_specific = QRadioButton("Convert All Data to Specific Timezone")

        # Create and configure the entry for specific timezone input
        self.specific_timezone_input = QLineEdit()
        self.specific_timezone_input.setPlaceholderText("Enter timezone (e.g. UTC)")
        self.specific_timezone_input.setToolTip("Enter a valid timezone (e.g. UTC, UTC-06:00, US/Pacific, Europe/London)")
        self.specific_timezone_input.setEnabled(False)
        self.specific_timezone_input.setFixedWidth(150)

        # Create and configure the button group for timezone handling options
        self.timezone_button_group = QButtonGroup()
        self.timezone_button_group.addButton(self.radio_remove_nonprimary, 0)
        self.timezone_button_group.addButton(self.radio_convert_to_primary, 1)
        self.timezone_button_group.addButton(self.radio_convert_to_local, 2)
        self.timezone_button_group.addButton(self.radio_convert_to_specific, 3)

        self.radio_convert_to_specific.toggled.connect(self.specific_timezone_input.setEnabled)

        self.radio_remove_nonprimary.setChecked(True)

        timezone_handling_layout.addStretch(1)
        timezone_handling_layout.addWidget(self.radio_remove_nonprimary)
        timezone_handling_layout.addWidget(self.radio_convert_to_primary)
        timezone_handling_layout.addWidget(self.radio_convert_to_local)
        timezone_handling_layout.addWidget(self.radio_convert_to_specific)
        timezone_handling_layout.addWidget(self.specific_timezone_input)
        timezone_handling_layout.addStretch(1)

        timezone_handling_group.setLayout(timezone_handling_layout)
        return timezone_handling_group

    def _create_additional_options_group(self) -> QGroupBox:
        """
        Create the additional options group.

        This method creates a group box containing additional options for the preprocessor.

        Returns:
            QGroupBox: The group box containing the additional options widgets.
        """
        LOGGER.debug("Creating additional options group")
        additional_options_group = QGroupBox("Additional Options")
        additional_options_layout = QHBoxLayout()

        # Checkbox for correcting duplicate event timestamps
        self.correct_duplicate_event_timestamps_checkbox = QCheckBox("Correct Duplicate Event Timestamps?")
        self.correct_duplicate_event_timestamps_checkbox.setChecked(True)
        additional_options_layout.addWidget(self.correct_duplicate_event_timestamps_checkbox, alignment=Qt.AlignCenter)  # type: ignore

        # Checkbox for using survey data (currently disabled)
        self.use_survey_data_checkbox = QCheckBox("Use Survey Data?")
        self.use_survey_data_checkbox.stateChanged.connect(self._survey_data_checkbox_update)
        self.use_survey_data_checkbox.setDisabled(True)
        additional_options_layout.addWidget(self.use_survey_data_checkbox, alignment=Qt.AlignCenter)  # type: ignore

        additional_options_group.setLayout(additional_options_layout)
        return additional_options_group

    def _create_survey_data_folder_group(self) -> QGroupBox:
        """
        Create the survey data folder selection group.

        This method creates a group box containing widgets for selecting the survey data folder.

        Returns:
            QGroupBox: The group box containing the survey data folder selection widgets.
        """
        LOGGER.debug("Creating survey data folder selection group")
        survey_data_folder_group = QGroupBox("Survey Data Folder Selection")
        survey_data_folder_layout = QVBoxLayout()

        # Button for selecting the survey data folder
        button_layout = QHBoxLayout()
        self.select_survey_data_folder_button = QPushButton("Select Survey Data Folder")
        self.select_survey_data_folder_button.clicked.connect(self._select_and_validate_raw_data_folder)
        self.select_survey_data_folder_button.setStyleSheet("QPushButton { padding: 10px; }")

        button_layout.addStretch()
        button_layout.addWidget(self.select_survey_data_folder_button)
        button_layout.addStretch()

        survey_data_folder_layout.addLayout(button_layout)
        survey_data_folder_layout.addSpacing(10)

        # Label for the survey data folder
        label_layout = QHBoxLayout()
        self.survey_data_folder_label = QLabel("Select the folder containing the Chronicle Android survey data")
        self.survey_data_folder_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.survey_data_folder_label.setWordWrap(True)
        self.survey_data_folder_label.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; padding: 5px; border-radius: 4px; }")

        label_layout.addStretch()
        label_layout.addWidget(self.survey_data_folder_label)
        label_layout.addStretch()

        survey_data_folder_layout.addLayout(label_layout)
        survey_data_folder_layout.addSpacing(10)

        survey_data_folder_group.setLayout(survey_data_folder_layout)
        survey_data_folder_group.setVisible(False)
        return survey_data_folder_group

    def _create_bottom_buttons(self) -> QHBoxLayout:
        """
        Create the bottom buttons group.

        This method creates a horizontal layout containing the bottom buttons for the UI.

        Returns:
            QHBoxLayout: The horizontal layout containing the bottom buttons.
        """
        LOGGER.debug("Creating bottom buttons group")
        bottom_button_layout = QHBoxLayout()

        # Button to display all interaction types
        self.display_interaction_type_window_button = QPushButton("Show All Interaction Types")
        self.display_interaction_type_window_button.setStyleSheet("QPushButton { padding: 10px; }")
        self.display_interaction_type_window_button.clicked.connect(lambda: self._display_interaction_type_window(mode="all"))
        bottom_button_layout.addWidget(self.display_interaction_type_window_button, alignment=Qt.AlignCenter)  # type: ignore

        # Button to run the preprocessor
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run)
        self.run_button.setStyleSheet("QPushButton { padding: 10px; }")
        bottom_button_layout.addWidget(self.run_button, alignment=Qt.AlignCenter)  # type: ignore

        # Button to open the output folder
        self.open_output_folder_button = QPushButton("Open Output Folder")
        self.open_output_folder_button.clicked.connect(self._open_output_folder)
        self.open_output_folder_button.setStyleSheet("QPushButton { padding: 10px; }")
        bottom_button_layout.addWidget(self.open_output_folder_button, alignment=Qt.AlignCenter)

        return bottom_button_layout

    def _load_and_set_config(self) -> None:
        """
        Load and set the configuration for the application.

        This method loads the configuration settings from a JSON file and applies them to the
        preprocessor options and UI elements. If the configuration file is not found, it logs a warning.
        """
        LOGGER.debug("Loading and setting configuration")
        try:
            with Path("Chronicle_Android_raw_data_preprocessor_app_config.json").open("r") as f:
                config = json.load(f)
                LOGGER.debug("Configuration file loaded successfully.")
        except FileNotFoundError:
            LOGGER.warning("Configuration file not found")
            return

        # Set the raw data folder from the configuration
        self.preprocessor_options.raw_data_folder = config.get("raw_data_folder", "Select the folder containing the Chronicle Android raw data")
        self.raw_data_folder_label.setText(str(self.preprocessor_options.raw_data_folder))

        # Set the filter file from the configuration
        self.preprocessor_options.filter_file = config.get("filter_file", "Select the file containing the apps to filter")
        self.filter_file_label.setText(str(self.preprocessor_options.filter_file))

        # Set the minimum usage duration from the configuration
        self.minimum_usage_duration_entry.setText(str(config.get("minimum_usage_duration", self.preprocessor_options.minimum_usage_duration)))

        # Set the custom app engagement duration from the configuration
        self.preprocessor_options.custom_app_engagement_duration = int(
            config.get("custom_app_engagement_duration", self.preprocessor_options.custom_app_engagement_duration)
        )

        # Set the correct duplicate event timestamps option from the configuration
        self.preprocessor_options.correct_duplicate_event_timestamps = config.get(
            "correct_duplicate_event_timestamps", self.preprocessor_options.correct_duplicate_event_timestamps
        )

        # Set the interaction types to stop usage at from the configuration
        self.preprocessor_options.other_interaction_types_to_stop_usage_at = set(
            config.get("interaction_types_to_stop_usage_at", self.preprocessor_options.other_interaction_types_to_stop_usage_at)
        )

        # Set the interaction types to remove from the configuration
        self.preprocessor_options.interaction_types_to_remove = set(
            config.get("interaction_types_to_remove", self.preprocessor_options.interaction_types_to_remove)
        )

        LOGGER.debug("Configuration settings applied successfully.")

    def _save_config(self) -> None:
        """
        Save the current configuration of the application.

        This method saves the current preprocessor options to a JSON file. It converts values to
        JSON serializable formats if necessary.
        """
        LOGGER.debug("Saving configuration")

        def convert_value(value):
            """
            Convert a value to a JSON serializable format.

            Args:
                value: The value to convert.

            Returns:
                The converted value.
            """
            if self.is_json_serializable(value):
                return value
            elif isinstance(value, typing.Iterable) and not isinstance(value, (str, bytes)):
                return list(value)
            else:
                return str(value)

        try:
            # Open the configuration file for writing
            with open("Chronicle_Android_raw_data_preprocessor_app_config.json", "w") as f:
                # Convert the preprocessor options to a dictionary and save it as JSON
                combined_dict = {k: convert_value(v) for k, v in self.preprocessor_options.__dict__.items()}
                json.dump(combined_dict, f, indent=4)
                LOGGER.debug("Configuration saved successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to save configuration: {e}")

    @typing.no_type_check
    def _on_worker_finished(self):
        """
        Handle the completion of the worker thread.

        This method is called when the worker thread finishes processing. It saves the configuration,
        brings the main window to the front, and displays a message box indicating that the processing
        is finished.
        """
        LOGGER.debug(f"{self.worker.__class__.__name__} finished processing.")

        # Save the current configuration
        self._save_config()

        # Bring the main window to the front
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

        # Display a message box indicating that the processing is finished
        msg_box = QMessageBox(QMessageBox.Information, "Finished", "Finished preprocessing Chronicle Android raw data.", QMessageBox.Ok, self)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
        msg_box.raise_()
        msg_box.activateWindow()
        msg_box.exec_()

        # Enable the run button
        self.run_button.setEnabled(True)

        # Reset the window flags
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

        # Delete the worker instance
        del self.worker

    def _on_worker_error(self, error_message: str) -> None:
        """
        Handle errors encountered by the worker thread.

        Args:
            error_message (str): The error message to display.

        This method is called when the worker thread encounters an error. It logs the error and
        displays a message box with the error message.
        """
        LOGGER.error(f"Worker encountered an error: {error_message}")

        # Bring the main window to the front
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

        # Display a message box with the error message
        msg_box = QMessageBox(QMessageBox.Warning, "Error", error_message, QMessageBox.Ok, self)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
        msg_box.raise_()
        msg_box.activateWindow()
        msg_box.exec_()

        # Enable the run button
        self.run_button.setEnabled(True)

        # Reset the window flags
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

        # Delete the worker instance
        del self.worker

    def _run(self) -> None:
        """
        Start the preprocessing worker thread.

        This method initializes the preprocessor options, creates a worker thread, and starts it.
        It also disables the run button to prevent multiple workers from running simultaneously.
        """
        # Check if a worker is already running
        if hasattr(self, "worker") and self.worker.isRunning():
            LOGGER.warning("Attempted to start a new worker while another is already running.")
            QMessageBox.warning(self, "Warning", "A worker is already running.")
            return

        # Set default values for minimum usage duration and custom app engagement duration if they are empty
        if self.minimum_usage_duration_entry.text().strip() == "":
            self.minimum_usage_duration_entry.setText("0")
        self.preprocessor_options.minimum_usage_duration = int(self.minimum_usage_duration_entry.text().strip())

        if self.custom_app_engagement_duration_entry.text().strip() == "":
            self.custom_app_engagement_duration_entry.setText("300")
        self.preprocessor_options.custom_app_engagement_duration = int(self.custom_app_engagement_duration_entry.text().strip())

        # Set the timezone handling option
        self.preprocessor_options.timezone_handling_option = TimezoneHandlingOption(self.timezone_button_group.checkedId())

        # Set the specific timezone
        self.preprocessor_options.specific_timezone = self.specific_timezone_input.text().strip()

        # Set the correct duplicate event timestamps option
        self.preprocessor_options.correct_duplicate_event_timestamps = self.correct_duplicate_event_timestamps_checkbox.isChecked()

        # Create and start the worker thread
        self.worker = PreprocessorThreadWorker(
            self.preprocessor_options,
        )
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()
        LOGGER.debug("Worker started.")

        # Disable the run button
        self.run_button.setEnabled(False)

    @typing.override
    def closeEvent(self, event) -> None:  # type: ignore
        """
        Handle the window close event.

        Args:
            event: The close event.

        This method is called when the user attempts to close the window. It prevents the window
        from closing if the worker thread is still running.
        """
        # Check if the worker thread is running
        if hasattr(self, "worker") and self.worker.isRunning():
            LOGGER.warning("Attempted to close the window while the worker is running.")
            QMessageBox.warning(self, "Warning", "Please do not close the window while the script is running.")
            event.ignore()
        else:
            LOGGER.debug("Window closed.")
            event.accept()


if __name__ == "__main__":
    """
    Main entry point for the Chronicle Android Raw Data Preprocessor application.

    This script initializes the application, sets up logging, and starts the GUI.
    """
    ALL_INTERACTION_TYPES_MAP = {
        "Instance of Usage for an App": InteractionType.APP_USAGE,
        "Activity Resumed for a Filtered App": InteractionType.FILTERED_APP_RESUMED,
        "Activity Paused for a Filtered App": InteractionType.FILTERED_APP_PAUSED,
        "Instance of Usage for a Filtered App": InteractionType.FILTERED_APP_USAGE,
        "Missing End of Usage after an App Starts Being Used": InteractionType.END_OF_USAGE_MISSING,
        "Unknown importance: 1": InteractionType.ACTIVITY_RESUMED,
        "Unknown importance: 2": InteractionType.ACTIVITY_PAUSED,
        "Unknown importance: 3": InteractionType.END_OF_DAY,
        "Unknown importance: 4": InteractionType.CONTINUE_PREVIOUS_DAY,
        "Unknown importance: 5": InteractionType.CONFIGURATION_CHANGE,
        "Unknown importance: 6": InteractionType.SYSTEM_INTERACTION,
        "Unknown importance: 7": InteractionType.USER_INTERACTION,
        "Unknown importance: 8": InteractionType.SHORTCUT_INVOCATION,
        "Unknown importance: 9": InteractionType.CHOOSER_ACTION,
        "Unknown importance: 10": InteractionType.NOTIFICATION_SEEN,
        "Unknown importance: 11": InteractionType.STANDBY_BUCKET_CHANGED,
        "Unknown importance: 12": InteractionType.NOTIFICATION_INTERRUPTION,
        "Unknown importance: 13": InteractionType.SLICE_PINNED_PRIV,
        "Unknown importance: 14": InteractionType.SLICE_PINNED_APP,
        "Unknown importance: 15": InteractionType.SCREEN_INTERACTIVE,
        "Unknown importance: 16": InteractionType.SCREEN_NON_INTERACTIVE,
        "Unknown importance: 17": InteractionType.KEYGUARD_SHOWN,
        "Unknown importance: 18": InteractionType.KEYGUARD_HIDDEN,
        "Unknown importance: 19": InteractionType.FOREGROUND_SERVICE_START,
        "Unknown importance: 20": InteractionType.FOREGROUND_SERVICE_STOP,
        "Unknown importance: 21": InteractionType.CONTINUING_FOREGROUND_SERVICE,
        "Unknown importance: 22": InteractionType.ROLLOVER_FOREGROUND_SERVICE,
        "Unknown importance: 23": InteractionType.ACTIVITY_STOPPED,
        "Unknown importance: 24": InteractionType.ACTIVITY_DESTROYED,
        "Unknown importance: 25": InteractionType.FLUSH_TO_DISK,
        "Unknown importance: 26": InteractionType.DEVICE_SHUTDOWN,
        "Unknown importance: 27": InteractionType.DEVICE_STARTUP,
        "Unknown importance: 28": InteractionType.USER_UNLOCKED,
        "Unknown importance: 29": InteractionType.USER_STOPPED,
        "Unknown importance: 30": InteractionType.LOCUS_ID_SET,
        "Unknown importance: 31": InteractionType.APP_COMPONENT_USED,
        "Move to Foreground": InteractionType.ACTIVITY_RESUMED,
        "Move to Background": InteractionType.ACTIVITY_PAUSED,
    }

    POSSIBLE_INTERACTION_TYPES_TO_REMOVE = {
        "Instance of Usage for a Filtered App": InteractionType.FILTERED_APP_USAGE,
        "Missing End of Usage after an App Starts Being Used": InteractionType.END_OF_USAGE_MISSING,
        "Unknown importance: 3": InteractionType.END_OF_DAY,
        "Unknown importance: 4": InteractionType.CONTINUE_PREVIOUS_DAY,
        "Unknown importance: 5": InteractionType.CONFIGURATION_CHANGE,
        "Unknown importance: 6": InteractionType.SYSTEM_INTERACTION,
        "Unknown importance: 7": InteractionType.USER_INTERACTION,
        "Unknown importance: 8": InteractionType.SHORTCUT_INVOCATION,
        "Unknown importance: 9": InteractionType.CHOOSER_ACTION,
        "Unknown importance: 10": InteractionType.NOTIFICATION_SEEN,
        "Unknown importance: 11": InteractionType.STANDBY_BUCKET_CHANGED,
        "Unknown importance: 12": InteractionType.NOTIFICATION_INTERRUPTION,
        "Unknown importance: 13": InteractionType.SLICE_PINNED_PRIV,
        "Unknown importance: 14": InteractionType.SLICE_PINNED_APP,
        "Unknown importance: 15": InteractionType.SCREEN_INTERACTIVE,
        "Unknown importance: 16": InteractionType.SCREEN_NON_INTERACTIVE,
        "Unknown importance: 17": InteractionType.KEYGUARD_SHOWN,
        "Unknown importance: 18": InteractionType.KEYGUARD_HIDDEN,
        "Unknown importance: 19": InteractionType.FOREGROUND_SERVICE_START,
        "Unknown importance: 20": InteractionType.FOREGROUND_SERVICE_STOP,
        "Unknown importance: 21": InteractionType.CONTINUING_FOREGROUND_SERVICE,
        "Unknown importance: 22": InteractionType.ROLLOVER_FOREGROUND_SERVICE,
        "Unknown importance: 23": InteractionType.ACTIVITY_STOPPED,
        "Unknown importance: 24": InteractionType.ACTIVITY_DESTROYED,
        "Unknown importance: 25": InteractionType.FLUSH_TO_DISK,
        "Unknown importance: 26": InteractionType.DEVICE_SHUTDOWN,
        "Unknown importance: 27": InteractionType.DEVICE_STARTUP,
        "Unknown importance: 28": InteractionType.USER_UNLOCKED,
        "Unknown importance: 29": InteractionType.USER_STOPPED,
        "Unknown importance: 30": InteractionType.LOCUS_ID_SET,
        "Unknown importance: 31": InteractionType.APP_COMPONENT_USED,
    }

    POSSIBLE_SAME_APP_INTERACTION_TYPES_TO_STOP_USAGE_AT = {
        "Unknown importance: 2\n Activity Paused for the Same App": InteractionType.ACTIVITY_PAUSED,
        "Unknown importance: 23\n Activity Stopped for the Same App": InteractionType.ACTIVITY_STOPPED,
        "Unknown importance: 24\n Activity Destroyed for the Same App": InteractionType.ACTIVITY_DESTROYED,
    }

    POSSIBLE_OTHER_INTERACTION_TYPES_TO_STOP_USAGE_AT = {
        "Unknown importance: 1\n Activity Resumed for a Different App": InteractionType.ACTIVITY_RESUMED,
        "Unknown importance: 16": InteractionType.SCREEN_NON_INTERACTIVE,
        "Unknown importance: 17": InteractionType.KEYGUARD_SHOWN,
        "Unknown importance: 24": InteractionType.ACTIVITY_DESTROYED,
        "Unknown importance: 26": InteractionType.DEVICE_SHUTDOWN,
        "Unknown importance: 29": InteractionType.USER_STOPPED,
        "Activity Resumed for a Filtered App": InteractionType.FILTERED_APP_RESUMED,
        "Instance of Usage for a Filtered App": InteractionType.FILTERED_APP_USAGE,
    }

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - Line %(lineno)d - %(message)s",
        handlers=[logging.FileHandler("Chronicle_Android_raw_data_preprocessor_app.log"), logging.StreamHandler()],
    )

    # Create a logger instance
    LOGGER = logging.getLogger(__name__)

    # Initialize the application
    app = QApplication(sys.argv)
    ex = ChronicleAndroidRawDataPreprocessorGUI()
    ex.show()
    # Start the application event loop
    sys.exit(app.exec_())
