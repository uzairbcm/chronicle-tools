from __future__ import annotations

import asyncio
import datetime
import json
import logging
import shutil
import sys
import traceback
import typing
from datetime import datetime as datetime_class
from datetime import tzinfo
from enum import StrEnum
from pathlib import Path

import aiofiles
import httpx
import regex as re
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class FilterType(StrEnum):
    """
    Enum for filter types used in the application.

    Attributes:
        INCLUSIVE (str): Represents an inclusive filter.
        EXCLUSIVE (str): Represents an exclusive filter.
    """

    INCLUSIVE = "Inclusive"
    EXCLUSIVE = "Exclusive"


class ChronicleDeviceType(StrEnum):
    """
    Enum for different types of devices supported by Chronicle.

    Attributes:
        AMAZON (str): Represents an Amazon Fire device.
        ANDROID (str): Represents an Android device.
        IPHONE (str): Represents an iPhone device.
    """

    AMAZON = "Amazon Fire"
    ANDROID = "Android"
    IPHONE = "iPhone"


class ChronicleDataType(StrEnum):
    """
    Enum for different types of data collected by Chronicle.

    Attributes:
        RAW (str): Represents raw usage events data.
        SURVEY (str): Represents app usage survey data.
        IOSSENSOR (str): Represents iOS sensor data.
    """

    RAW = "UsageEvents"
    SURVEY = "AppUsageSurvey"
    IOSSENSOR = "IOSSensor"


def get_matching_files_from_folder(
    folder: Path | str,
    file_matching_pattern: str,
    ignore_names: list[str] | None = None,
) -> list[Path]:
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
        ignore_names = []
    matching_files = [
        Path(f)
        for f in Path(folder).rglob(r"**")
        if Path(f).is_file() and re.search(file_matching_pattern, str(f)) and all(ignored not in str(f) for ignored in ignore_names)
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
    return datetime_class.now(datetime.timezone.utc).astimezone().tzinfo
    # logger.debug(f"Local timezone determined: {local_timezone}")


class DownloadThreadWorker(QThread):
    """
    A worker thread for downloading Chronicle Android bulk data.

    Signals:
        finished: Emitted when the download is complete.
        error: Emitted when an error occurs during the download, with the error message as a string.

    Attributes:
        parent_ (ChronicleAndroidBulkDataDownloader): The parent downloader instance.
    """

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, parent_: ChronicleAndroidBulkDataDownloader) -> None:
        """
        Initializes the DownloadThreadWorker.

        Args:
            parent_ (ChronicleAndroidBulkDataDownloader): The parent downloader instance.
        """
        super().__init__(parent_)
        self.parent_ = parent_

    def run(self) -> None:
        """
        Runs the download process in a separate thread.
        """
        try:
            self._run()
        except Exception as e:
            self.error.emit(traceback.format_exc())

    def _run(self):
        """
        The main logic for downloading the data. Checks for valid inputs and handles the download process.
        Emits error signals if any issues are encountered.
        """
        if self.parent_.download_folder == "" or self.parent_.download_folder is None:
            self.error.emit("Please select a download folder.")
            LOGGER.warning("No download folder selected")
            return

        # expected_auth_token_length = 1130
        # if len(self.parent_.authorization_token_entry.toPlainText().strip()) < expected_auth_token_length:
        #     self.error.emit("Please enter a valid authorization token.")
        #     LOGGER.warning("Invalid authorization token entered")
        #     return

        expected_study_id_length = 36
        if len(self.parent_.study_id_entry.text().strip()) < expected_study_id_length:
            self.error.emit("Please enter a valid Chronicle study ID.")
            LOGGER.warning("Invalid study ID entered")
            return

        if (
            self.parent_.inclusive_filter_checkbox.isChecked() and len(self.parent_.participant_ids_to_filter_list_entry.toPlainText()) < 1
        ) or self.parent_.participant_ids_to_filter_list_entry.toPlainText() is None:
            self.error.emit("Please enter a valid list of participant IDs to *include* when the *inclusive* list checkbox is checked.")
            LOGGER.warning("Invalid participant IDs list entered for inclusive filter")
            return

        try:
            asyncio.run(self.parent_.download_participant_Chronicle_data_from_study())
        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code
            match error_code:
                case 401:
                    description = "Unauthorized. Please check the authorization token and try again."
                case 403:
                    description = "Forbidden"
                case 404:
                    description = "Not Found"
                case _:
                    description = "Unknown"

            LOGGER.exception(f"HTTP error occurred: {error_code} {description}")
            self.error.emit(f"An HTTP error occurred while attempting to download the data:\n\n{error_code} {description}")
            return
        except Exception as e:
            LOGGER.exception("An error occurred while downloading the data")
            self.error.emit(f"An error occurred while downloading the data: {traceback.format_exc()}")
            return
        else:
            self.parent_.archive_downloaded_data()
            self.parent_.organize_downloaded_data()
            with Path("Chronicle_Android_bulk_data_downloader_config.json").open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "download_folder": self.parent_.download_folder,
                            "study_id": self.parent_.study_id_entry.text().strip(),
                            "participant_ids_to_filter": self.parent_.participant_ids_to_filter_list_entry.toPlainText(),
                            "inclusive_checked": self.parent_.inclusive_filter_checkbox.isChecked(),
                            "survey_checked": self.parent_.download_survey_data_checkbox.isChecked(),
                        }
                    )
                )
            LOGGER.debug("Data download complete")
            self.finished.emit()


class ChronicleAndroidBulkDataDownloader(QWidget):
    """
    A QWidget-based application for downloading bulk data from Chronicle Android.

    Attributes:
        download_folder (Path | str): The folder where downloaded files will be saved.
        temp_download_file_pattern (str): Regex pattern for temporary download files.
        dated_file_pattern (str): Regex pattern for dated files.
        raw_data_file_pattern (str): Regex pattern for raw data files.
        survey_data_file_pattern (str): Regex pattern for survey data files.
    """

    def __init__(self) -> None:
        """
        Initializes the ChronicleAndroidBulkDataDownloader class.
        """
        super().__init__()
        self.download_folder: Path | str = ""
        self.temp_download_file_pattern: str = r"[\s\S]*.csv"
        self.dated_file_pattern: str = r"([\s\S]*(\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4})[\s\S]*.csv)"
        self.raw_data_file_pattern: str = r"[\s\S]*(Raw)[\s\S]*.csv"
        self.survey_data_file_pattern: str = r"[\s\S]*(Survey)[\s\S]*.csv"
        self._init_UI()
        self._load_and_set_config()

    def _select_and_validate_download_folder(self) -> None:
        """
        Select and validate the download folder.

        This method opens a file dialog for the user to select the download folder and validates
        the selected folder.
        """
        LOGGER.debug("Selecting download folder")
        current_download_folder_label = self.download_folder_label.text().strip()
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Download Folder")

        if selected_folder and Path(selected_folder).is_dir():
            self.download_folder = selected_folder
            self.download_folder_label.setText(selected_folder)
            LOGGER.debug(f"Selected download folder: {selected_folder}")
        else:
            self.download_folder_label.setText(current_download_folder_label)
            LOGGER.debug("Invalid folder selected or no folder selected, reset to previous value")

        self.adjustSize()

    def _update_list_label_text(self) -> None:
        """
        Updates the label text based on the state of the inclusive filter checkbox.
        """
        if self.inclusive_filter_checkbox.isChecked():
            self.list_ids_label.setText("List of participant IDs to *include* (separated by commas):")
        else:
            self.list_ids_label.setText("List of participant IDs to *exclude* (separated by commas):")
        LOGGER.debug("Updated label text based on inclusive filter checkbox state")

        self.adjustSize()

    def _init_UI(self) -> None:
        """
        Initializes the user interface.
        """
        LOGGER.debug("Initializing UI")
        self.setWindowTitle("Chronicle Android Bulk Data Downloader")
        self.setGeometry(100, 100, 500, 350)

        main_layout = QVBoxLayout()

        # Add folder selection group
        main_layout.addWidget(self._create_folder_selection_group())
        main_layout.addSpacing(10)

        # Add token entry group
        main_layout.addWidget(self._create_authorization_token_entry_group())
        main_layout.addSpacing(10)

        # Add study ID entry group
        main_layout.addWidget(self._create_study_id_entry_group())
        main_layout.addSpacing(10)

        # Add participant IDs entry group
        main_layout.addWidget(self._create_participant_ids_entry_group())
        main_layout.addSpacing(10)

        # Add checkbox layout
        main_layout.addLayout(self._create_checkbox_layout())
        main_layout.addSpacing(10)

        # Add button layout
        main_layout.addLayout(self._create_button_layout())
        main_layout.addSpacing(10)

        self.setLayout(main_layout)
        self._center_window()
        self.adjustSize()
        LOGGER.debug("Initialized UI")

    def _create_folder_selection_group(self) -> QGroupBox:
        """
        Creates the folder selection group box.
        """
        group_box = QGroupBox("Folder Selection")
        group_layout = QVBoxLayout()

        # Add button for selecting download folder
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.select_download_folder_button = QPushButton("Select Download Folder")
        self.select_download_folder_button.clicked.connect(self._select_and_validate_download_folder)
        self.select_download_folder_button.setStyleSheet("QPushButton { padding: 10px; }")
        button_layout.addWidget(self.select_download_folder_button)
        button_layout.addStretch()
        group_layout.addLayout(button_layout)

        # Add label for download folder
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.download_folder_label = QLabel("Select the folder to download the Chronicle Android raw data to")
        self.download_folder_label.setStyleSheet(
            """QLabel {
                font-size: 10pt;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
            }"""
        )
        self.download_folder_label.setWordWrap(True)
        self.download_folder_label.setAlignment(Qt.AlignCenter) # type: ignore
        self.download_folder_label.setFixedWidth(400)
        label_layout.addStretch()
        label_layout.addWidget(self.download_folder_label)
        label_layout.addStretch()
        group_layout.addLayout(label_layout)

        group_box.setLayout(group_layout)
        return group_box

    def _create_authorization_token_entry_group(self) -> QGroupBox:
        """
        Creates the authorization token entry group box.
        """
        group_box = QGroupBox("Authorization Token Entry")
        group_layout = QVBoxLayout()

        # Add label for token entry
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.authorization_token_label = QLabel("Please paste the temporary authorization token:")
        self.authorization_token_label.setWordWrap(True)
        self.authorization_token_label.setFixedWidth(250)
        label_layout.addWidget(self.authorization_token_label)
        label_layout.addStretch()
        group_layout.addLayout(label_layout)

        # Add text edit for token entry
        entry_layout = QHBoxLayout()
        entry_layout.addStretch()
        self.authorization_token_entry = QTextEdit()
        self.authorization_token_entry.setFixedSize(300, 75)
        self.authorization_token_entry.setStyleSheet("""
            QTextEdit {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QTextEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        entry_layout.addWidget(self.authorization_token_entry)
        entry_layout.addStretch()
        group_layout.addLayout(entry_layout)

        group_box.setLayout(group_layout)
        return group_box

    def _create_study_id_entry_group(self) -> QGroupBox:
        """
        Creates the study ID entry group box.
        """
        group_box = QGroupBox("Study ID Entry")
        group_layout = QVBoxLayout()

        # Add label for study ID entry
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.study_id_label = QLabel("Please paste the study ID:")
        label_layout.addWidget(self.study_id_label)
        label_layout.addStretch()
        group_layout.addLayout(label_layout)

        # Add line edit for study ID entry
        entry_layout = QHBoxLayout()
        entry_layout.addStretch()
        self.study_id_entry = QLineEdit()
        self.study_id_entry.setFixedWidth(226)
        self.study_id_entry.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        entry_layout.addWidget(self.study_id_entry)
        entry_layout.addStretch()
        group_layout.addLayout(entry_layout)

        group_box.setLayout(group_layout)
        return group_box

    def _create_participant_ids_entry_group(self) -> QGroupBox:
        """
        Creates the participant IDs entry group box.
        """
        group_box = QGroupBox("Participant IDs Entry")
        group_layout = QVBoxLayout()

        # Add label for participant IDs entry
        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.list_ids_label = QLabel("List of participant IDs to *exclude* (separated by commas):")
        label_layout.addWidget(self.list_ids_label)
        label_layout.addStretch()
        group_layout.addLayout(label_layout)

        # Add checkbox for inclusive filter
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()
        self.inclusive_filter_checkbox = QCheckBox("Use *Inclusive* List Instead")
        self.inclusive_filter_checkbox.stateChanged.connect(self._update_list_label_text)
        checkbox_layout.addWidget(self.inclusive_filter_checkbox)
        checkbox_layout.addStretch()
        group_layout.addLayout(checkbox_layout)

        # Add text edit for participant IDs entry
        entry_layout = QHBoxLayout()
        entry_layout.addStretch()
        self.participant_ids_to_filter_list_entry = QTextEdit()
        self.participant_ids_to_filter_list_entry.setFixedSize(300, 100)
        self.participant_ids_to_filter_list_entry.setStyleSheet("""
            QTextEdit {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QTextEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        entry_layout.addWidget(self.participant_ids_to_filter_list_entry)
        entry_layout.addStretch()
        group_layout.addLayout(entry_layout)

        group_box.setLayout(group_layout)
        return group_box

    def _center_window(self):
        """
        Centers the application window on the screen.
        """
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())  # type: ignore
        centerPoint = QApplication.desktop().screenGeometry(screen).center()  # type: ignore
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
        LOGGER.debug("Centered the window")

    def _create_checkbox_layout(self) -> QHBoxLayout:
        """
        Creates the layout for the checkbox.
        """
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()
        self.download_survey_data_checkbox = QCheckBox("Download Survey Data")
        checkbox_layout.addWidget(self.download_survey_data_checkbox)
        checkbox_layout.addStretch()
        return checkbox_layout

    def _create_button_layout(self) -> QHBoxLayout:
        """
        Creates the layout for the button.
        """
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run)
        self.run_button.setStyleSheet("QPushButton { padding: 10px; }")
        button_layout.addWidget(self.run_button)
        button_layout.addStretch()
        return button_layout

    def _load_and_set_config(self) -> None:
        """
        Loads and sets the configuration from a JSON file.
        """
        try:
            with Path("Chronicle_Android_bulk_data_downloader_config.json").open("r") as f:
                config = json.load(f)
            LOGGER.debug("Loaded configuration from file")
        except FileNotFoundError:
            LOGGER.warning("Configuration file not found")
            return

        self.download_folder = config.get("download_folder", "")
        self.study_id_entry.setText(config.get("study_id", ""))
        self.participant_ids_to_filter_list_entry.setText(config.get("participant_ids_to_filter", ""))
        self.inclusive_filter_checkbox.setChecked(config.get("inclusive_checked", False))
        self.download_survey_data_checkbox.setChecked(config.get("survey_checked", False))

        if self.download_folder:
            self.download_folder_label.setText(str(self.download_folder))

        LOGGER.debug("Set configuration from loaded file")

    @staticmethod
    def delete_zero_byte_file(file: str | Path) -> None:
        """
        Deletes a zero-byte file.

        Args:
            file (str | Path): The file to delete.
        """
        if Path(file).stat().st_size == 0:
            try:
                Path(file).unlink()
                LOGGER.debug(f"Deleted zero-byte file: {file}")
            except PermissionError:
                LOGGER.exception(f"The 0 byte file {file} could not be removed due to already being open, please close it and try again.")

    def archive_downloaded_data(self) -> None:
        """
        Archives outdated downloaded data.
        """
        Chronicle_dated_files = get_matching_files_from_folder(
            folder=self.download_folder,
            file_matching_pattern=self.dated_file_pattern,
            ignore_names=["Archive", ".png"],
        )

        for file in Chronicle_dated_files:
            re_file_date = re.search(r"(\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4})", str(file))[0]  # type: ignore

            if not re_file_date:
                msg = f"File {file} possibly altered while script was running, please avoid doing this."
                LOGGER.error(msg)
                raise RuntimeError(msg)

            try:
                re_file_date_object = datetime_class.strptime(re_file_date, "%m-%d-%Y").replace(tzinfo=get_local_timezone())
            except ValueError:
                re_file_date_object = datetime_class.strptime(re_file_date, "%m.%d.%Y").replace(tzinfo=get_local_timezone())

            if re_file_date_object.date() < datetime_class.now(tz=get_local_timezone()).date():
                parent_dir_path = Path(file).parent
                parent_dir_name = Path(file).parent.name
                Path(f"{parent_dir_path}/{parent_dir_name} Archive/{parent_dir_name} Archive {re_file_date}").mkdir(parents=True, exist_ok=True)

                shutil.copy(
                    src=file,
                    dst=f"{parent_dir_path}/{parent_dir_name} Archive/{parent_dir_name} Archive {re_file_date}/{file.name}",
                )
                file.unlink()

        LOGGER.debug("Finished archiving outdated Chronicle Android data.")

    def organize_downloaded_data(self) -> None:
        """
        Organizes downloaded data into appropriate folders.
        """
        self.raw_data_folder = Path(self.download_folder) / "Chronicle Android Raw Data Downloads"
        self.survey_data_folder = Path(self.download_folder) / "Chronicle Android Survey Data Downloads"

        self.raw_data_folder.mkdir(parents=True, exist_ok=True)
        self.survey_data_folder.mkdir(parents=True, exist_ok=True)

        unorganized_raw_data_files = get_matching_files_from_folder(
            folder=self.download_folder,
            file_matching_pattern=self.raw_data_file_pattern,
            ignore_names=["Archive", "Chronicle Android Raw Data Downloads"],
        )

        for file in unorganized_raw_data_files:
            shutil.copy(src=file, dst=self.raw_data_folder)
            file.unlink()

        unorganized_survey_data_files = get_matching_files_from_folder(
            folder=self.download_folder,
            file_matching_pattern=self.survey_data_file_pattern,
            ignore_names=["Archive", "Chronicle Android Survey Data Downloads"],
        )

        for file in unorganized_survey_data_files:
            shutil.copy(src=file, dst=self.survey_data_folder)
            file.unlink()

        LOGGER.debug("Finished organizing downloaded Chronicle Android data.")

    def _exclusive_filter_participant_id_list(self, participant_id_list: list[str], participant_ids_to_filter: list[str]) -> list[str]:
        """
        Filters the participant ID list using an exclusive filter.

        Args:
            participant_id_list (list[str]): The list of participant IDs.
            participant_ids_to_filter (list[str]): The list of participant IDs to exclude.

        Returns:
            list[str]: The filtered list of participant IDs.
        """
        filtered_participant_id_list = [
            participant_id
            for participant_id in participant_id_list
            if participant_id is not None
            and not any(excluded_participant_id.lower() in participant_id.lower() for excluded_participant_id in participant_ids_to_filter)
        ]

        filtered_participant_id_list.sort()

        LOGGER.debug("Filtered participant ID list using exclusive filter")
        return filtered_participant_id_list

    def _inclusive_filter_participant_id_list(self, participant_id_list: list[str], participant_ids_to_filter: list[str]) -> list[str]:
        """
        Filters the participant ID list using an inclusive filter.

        Args:
            participant_id_list (list[str]): The list of participant IDs.
            participant_ids_to_filter (list[str]): The list of participant IDs to include.

        Returns:
            list[str]: The filtered list of participant IDs.
        """
        filtered_participant_id_list = [
            participant_id
            for participant_id in participant_id_list
            if participant_id is not None
            and any(included_participant_id.lower() in participant_id.lower() for included_participant_id in participant_ids_to_filter)
        ]

        filtered_participant_id_list.sort()

        LOGGER.debug("Filtered participant ID list using inclusive filter")
        return filtered_participant_id_list

    def _filter_participant_id_list(self, participant_id_list: list[str]) -> list[str]:
        """
        Filters the participant ID list based on the selected filter type.

        Args:
            participant_id_list (list[str]): The list of participant IDs.

        Returns:
            list[str]: The filtered list of participant IDs.
        """
        cleaned_participant_id_list = [pid.strip() for pid in participant_id_list if pid.strip()]

        participant_ids_to_filter_list = self.participant_ids_to_filter_list_entry.toPlainText().split(",")
        cleaned_participant_ids_to_filter_list = [pid.strip() for pid in participant_ids_to_filter_list if pid.strip()]

        if self.inclusive_filter_checkbox.isChecked():
            LOGGER.debug("Using inclusive filter for participant ID list")
            return self._inclusive_filter_participant_id_list(cleaned_participant_id_list, cleaned_participant_ids_to_filter_list)
        else:
            LOGGER.debug("Using exclusive filter for participant ID list")
            return self._exclusive_filter_participant_id_list(cleaned_participant_id_list, cleaned_participant_ids_to_filter_list)

    async def _download_participant_Chronicle_data_type(
        self,
        client: httpx.AsyncClient,
        participant_id: str,
        data_type: ChronicleDataType,
    ):
        """
        Downloads data of a specific type for a participant.

        Args:
            client (httpx.AsyncClient): The HTTP client to use for the request.
            participant_id (str): The participant ID.
            data_type (ChronicleDataType): The type of data to download.
        """
        semaphore = asyncio.Semaphore(1)
        async with semaphore:
            csv_response = await client.get(
                f"https://api.getmethodic.com/chronicle/v3/study/{self.study_id_entry.text().strip()}/participants/data?participantId={participant_id}&dataType={data_type}&fileType=csv",
                headers={"Authorization": f"Bearer {self.authorization_token_entry.toPlainText().strip()}"},
                timeout=60,
            )

        csv_response.raise_for_status()

        if data_type == ChronicleDataType.RAW:
            data_type_str = "Raw Data"
        elif data_type == ChronicleDataType.SURVEY:
            data_type_str = "Survey Data"
        else:
            raise ValueError

        output_filepath = (
            Path(self.download_folder)
            / f"{participant_id} Chronicle Android {data_type_str} {datetime_class.now(get_local_timezone()).strftime('%m-%d-%Y')}.csv"
        )
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(output_filepath, "wb") as f:
            await f.write(csv_response.content)

        LOGGER.debug(f"Downloaded {data_type_str} for participant {participant_id}")

        await asyncio.sleep(3)

    async def download_participant_Chronicle_data_from_study(self) -> None:
        """
        Downloads data for all participants in the study.

        This method retrieves the participant statistics from the Chronicle API,
        filters the participant IDs based on the selected filter type, and downloads
        the data for each participant.
        """
        client = httpx.AsyncClient(http2=True)

        participant_stats = await client.get(
            f"https://api.getmethodic.com/chronicle/v3/study/{self.study_id_entry.text().strip()}/participants/stats",
            headers={"Authorization": f"Bearer {self.authorization_token_entry.toPlainText().strip()}"},
            timeout=60,
        )

        participant_stats.raise_for_status()

        participant_id_list = [item["participantId"] for item in participant_stats.json().values()]

        filtered_participant_id_list = self._filter_participant_id_list(participant_id_list)

        if len(filtered_participant_id_list) == 0:
            msg = "No valid participant IDs found after filtering."
            LOGGER.error(msg)
            raise ValueError(msg)

        for i, participant_id in enumerate(filtered_participant_id_list):
            semaphore = asyncio.Semaphore(1)
            async with semaphore:
                await self._download_participant_Chronicle_data_type(
                    client=client,
                    participant_id=participant_id,
                    data_type=ChronicleDataType.RAW,
                )
                LOGGER.debug(
                    f"Finished downloading {ChronicleDataType.RAW} data for device {participant_id} ({i + 1}/{len(filtered_participant_id_list)})"
                )
                if self.download_survey_data_checkbox.isChecked():
                    await self._download_participant_Chronicle_data_type(
                        client=client,
                        participant_id=participant_id,
                        data_type=ChronicleDataType.SURVEY,
                    )
                    LOGGER.debug(
                        f"Finished downloading {ChronicleDataType.SURVEY} data for device {participant_id} ({i + 1}/{len(filtered_participant_id_list)})"
                    )

    def _run(self):
        self.worker = DownloadThreadWorker(self)
        self.worker.finished.connect(self.on_download_complete)
        self.worker.error.connect(self.on_download_error)
        self.worker.start()
        self.run_button.setEnabled(False)

    @typing.no_type_check
    def on_download_complete(self):
        del self.worker

        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

        msg_box = QMessageBox(QMessageBox.Information, "Finished", "Finished downloading Chronicle Android raw data.", QMessageBox.Ok, self)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
        msg_box.raise_()
        msg_box.activateWindow()
        msg_box.exec_()

        self.run_button.setEnabled(True)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    @typing.no_type_check
    def on_download_error(self, error_message):
        del self.worker

        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

        msg_box = QMessageBox(QMessageBox.Critical, "Error", error_message, QMessageBox.Ok, self)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
        msg_box.raise_()
        msg_box.activateWindow()
        msg_box.exec_()

        self.run_button.setEnabled(True)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - Line %(lineno)d - %(message)s",
        handlers=[logging.FileHandler("Chronicle_Android_bulk_data_downloader_app.log"), logging.StreamHandler()],
    )

    LOGGER = logging.getLogger(__name__)

    app = QApplication(sys.argv)
    ex = ChronicleAndroidBulkDataDownloader()
    ex.show()
    sys.exit(app.exec_())
