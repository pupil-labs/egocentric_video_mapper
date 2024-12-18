import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal

logger = logging.getLogger(__name__)


class OffsetCalculator:
    """This class provides methods to estimate the time offset between two signals and calculate the Pearson correlation coefficient.

    Args:
        source (ndarray): Source signal.
        source_timestamps (ndarray): Timestamps corresponding to the source signal.
        destination (array-like): Destination signal.
        destination_timestamps (array-like): Timestamps corresponding to the destination signal.
        resampling_frequency (int, optional): Frequency (Hz) for resampling signals. Defaults to 100.

    Attributes:
        source_resampled (ndarray): Resampled source signal data to the specified frequency.
        source_resampled_timestamps (ndarray): Resampled timestamps corresponding to the source signal data.
        destination_resampled (ndarray): Resampled destination signal data  to the specified frequency..
        destination_resampled_timestamps (ndarray): Resampled timestamps corresponding to the destination signal data.
        time_offset (float or None): Last estimated time offset between the source and destination signals in seconds. Everytime the method estimate_time_offset is called, this attribute is updated.
    """

    def __init__(
        self, src, src_timestamps_sec, dst, dst_timestamps_sec, resampling_frequency=100
    ):
        self.src = np.asarray(src).flatten()
        self.src_ts_sec = np.asarray(src_timestamps_sec).flatten()
        assert len(self.src) == len(
            self.src_ts_sec
        ), "Source signal and timestamps must have the same length"

        self.dst = np.asarray(dst).flatten()
        self.dst_ts_sec = np.asarray(dst_timestamps_sec).flatten()
        assert len(self.dst) == len(
            self.dst_ts_sec
        ), "Destination signal and timestamps must have the same length"

        self.resampling_frequency = resampling_frequency
        self.src_resampled, self.src_resampled_timestamps = self._resample_signal(
            self.src, self.src_ts_sec, self.resampling_frequency
        )
        self.dst_resampled, self.dst_resampled_timestamps = self._resample_signal(
            self.dst, self.dst_ts_sec, self.resampling_frequency
        )

        self.time_offset = None

    @staticmethod
    def _resample_signal(signal, timestamps, resampling_frequency=100):
        resampled_timestamps = np.arange(
            timestamps[0], timestamps[-1], 1 / resampling_frequency
        )
        resampled_signal = scipy.interpolate.interp1d(
            timestamps, signal, "linear", bounds_error=False, fill_value="extrapolate"
        )(resampled_timestamps)
        return resampled_signal, resampled_timestamps

    def estimate_time_offset(
        self, source_start_time=None, source_end_time=None, plot=False
    ):
        """Estimates the time offset (in seconds) between the source and destination signals by cross-correlating them and finding the lag index that maximizes the cross-correlation.

        Args:
            source_start_time (float, optional): Start time of the source signal interval that is going to be used to estimate the time offset.  If not specified, the beginning of the source signal is assumed. Defaults to None.
            source_end_time (float, optional): End time of the source signal interval that is going to be used to estimate the time offset. If not specified, the estimation extends to the end of the source signal.Defaults to None.

        Returns:
            float: Estimated time offset of the source signal relative to the destination signal, in seconds.
            float: Pearson correlation coefficient of the aligned source and destination signals. A value closer to 1 indicates better alignment.
        """
        if source_start_time is not None and source_end_time is not None:
            assert (
                source_start_time < source_end_time
            ), f"Start time ({source_start_time} s) must be smaller than end time ({source_end_time} s)"

        start_index, end_index = self._obtain_indexes(
            self.src_resampled_timestamps, source_start_time, source_end_time
        )

        crosscorrelation, lag_indexes = self._cross_correlate(
            self.src_resampled[start_index:end_index], self.dst_resampled
        )
        sampling_offset = lag_indexes[np.argmax(crosscorrelation)]
        correlation_score = self._correlation_coefficient(
            sampling_offset, self.src_resampled[start_index:end_index]
        )

        self.time_offset = (
            (sampling_offset / self.resampling_frequency)
            - self.src_resampled_timestamps[start_index]
            + self.dst_resampled_timestamps[0]
        )

        logger.info(
            f"With the interval going from {self.src_resampled_timestamps[start_index]}s to {self.src_resampled_timestamps[-1 if end_index is None else end_index]}s of the src signal, the estimated time offset is: {self.time_offset} seconds (Pearson correlation: {correlation_score})"
        )
        if plot:
            self.plot_synchronized_signals(sampling_offset, start_index, end_index)
        return self.time_offset, correlation_score

    def _obtain_indexes(self, timestamps, start_time, end_time):
        """Obtains the start and end indexes of the timestamps array corresponding to the specified time range.

        Args:
            timestamps (ndarray): Array containing timestamps.
            start_time (float): Start time of the desired time range.
            end_time (float): End time of the desired time range.

        Returns:
            int: Index of the start time position in the timestamps array.
            int: Index of the end time position in the timestamps array.
        """
        start_index = (
            0 if start_time is None else np.searchsorted(timestamps, start_time)
        )
        end_index = None if end_time is None else np.searchsorted(timestamps, end_time)
        return start_index, end_index

    @staticmethod
    def _cross_correlate(src_signal, dst_signal):
        crosscorrelation = scipy.signal.correlate(dst_signal, src_signal, mode="full")
        lag_indexes = scipy.signal.correlation_lags(
            dst_signal.size, src_signal.size, mode="full"
        )
        return crosscorrelation, lag_indexes

    def _correlation_coefficient(self, sampling_offset, src_signal):
        src_overlap, dst_overlap, _ = self._obtain_overlapping_signals(
            sampling_offset, src_signal
        )
        return np.corrcoef(src_overlap, dst_overlap)[0, 1]

    def plot_synchronized_signals(
        self, sampling_offset, source_start_time=0, source_end_time=None
    ):
        src, dst, dst_ts = self._obtain_overlapping_signals(
            sampling_offset, self.src_resampled[source_start_time:source_end_time]
        )
        plt.figure(figsize=(25, 5))
        plt.plot(dst_ts, dst, label="Destination signal")
        plt.plot(dst_ts, src, label="Source signal aligned to destination timeline")
        plt.vlines(
            self.time_offset,
            min(src),
            max(src),
            colors="r",
            linestyles="--",
            label=f"Estimated time offset of source signal wrt destination signal ({self.time_offset:.2f} s)",
        )
        plt.xlabel("Destination signal time (s)")
        plt.legend()
        plt.show()

    def _obtain_overlapping_signals(self, sampling_offset, src_signal):
        """Returns resampled source and destination signals clipped to their overlapping region based on a sampling offset. Used to calculate the Pearson correlation coefficient.

        Args:
            sampling_offset (int): The displacement of the source signal with respect to the destination signal
            source_signal (ndarray): Source signal to be clipped. Might not be the full signal.

        Returns:
            tuple: A tuple containing 3 arrays:
                - The given source signal clipped to the overlapping region.
                - The destination signal clipped to the overlapping region.
                - The timestamps of the destination signal clipped to the overlapping region.
        """
        if sampling_offset < 0:
            src_trimmed = src_signal[abs(sampling_offset) :]
            dst_trimmed = self.dst_resampled
            dst_trimmed_ts = self.dst_resampled_timestamps
        else:
            src_trimmed = src_signal
            dst_trimmed = self.dst_resampled[sampling_offset:]
            dst_trimmed_ts = self.dst_resampled_timestamps[sampling_offset:]
        n_dst = len(dst_trimmed)
        n_src = len(src_trimmed)
        N = min(n_dst, n_src)
        return src_trimmed[:N], dst_trimmed[:N], dst_trimmed_ts[:N]
