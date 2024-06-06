import numpy as np
import scipy.signal
import scipy.interpolate
import pandas as pd


class OffsetCalculator():
    """ This class provides methods to estimate the time offset between two signals and calculate the Pearson correlation coefficient.

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

    def __init__(self, src, src_timestamps, dst, dst_timestamps, resampling_frequency=100):
        self.src = np.asarray(src).reshape(-1)
        self.src_timestamps = np.asarray(src_timestamps).reshape(-1)
        assert len(self.src) == len(
            self.src_timestamps), "Source signal and timestamps must have the same length"
        self.dst = np.asarray(dst).reshape(-1)
        self.dst_timestamps = np.asarray(dst_timestamps).reshape(-1)
        assert len(self.dst) == len(
            self.dst_timestamps), "Destination signal and timestamps must have the same length"
        self.resampling_frequency = resampling_frequency
        self.src_resampled, self.src_resampled_timestamps = self._resample_signal(
            self.src, self.src_timestamps, self.resampling_frequency)
        self.dst_resampled, self.dst_resampled_timestamps = self._resample_signal(
            self.dst, self.dst_timestamps, self.resampling_frequency)
        self.time_offset = None

    @staticmethod
    def _resample_signal(signal, timestamps, resampling_frequency=100):
        resampled_timestamps = np.arange(
            timestamps[0], timestamps[-1], 1/resampling_frequency)
        resampled_signal = scipy.interpolate.interp1d(
            timestamps, signal, 'linear', bounds_error=False, fill_value='extrapolate')(resampled_timestamps)
        return resampled_signal, resampled_timestamps

    def estimate_time_offset(self, source_start_time=None, source_end_time=None):
        """Estimates the time offset (in seconds) between the source and destination signals by cross-correlating them and finding the lag index that maximizes the cross-correlation. 

        Args:
            source_start_time (float, optional): Start time of the source signal interval that is going to be used to estimate the time offset.  If not specified, the beginning of the source signal is assumed. Defaults to None.
            source_end_time (float, optional): End time of the source signal interval that is going to be used to estimate the time offset. If not specified, the estimation extends to the end of the source signal.Defaults to None.

        Returns:
            float: Estimated time offset of the source signal relative to the destination signal, in seconds.
            float: Pearson correlation coefficient of the aligned source and destination signals. A value closer to 1 indicates better alignment.
        """
        if source_start_time is not None and source_end_time is not None:
            assert (source_start_time <
                    source_end_time), f"Start time ({source_start_time} s) must be smaller than end time ({source_end_time} s)"
        start_index, end_index = self._obtain_indexes(
            self.src_resampled_timestamps, source_start_time, source_end_time)
        crosscorrelation, lag_indexes = self._cross_correlate(
            self.src_resampled[start_index:end_index], self.dst_resampled)
        sampling_offset = lag_indexes[np.argmax(crosscorrelation)]
        correlation_score = self._correlation_coefficient(
            sampling_offset, self.src_resampled[start_index:end_index])
        self.time_offset = (sampling_offset / self.resampling_frequency) - \
            self.src_resampled_timestamps[start_index] + \
            self.dst_resampled_timestamps[0]
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
        start_index = 0 if start_time is None else np.searchsorted(timestamps, start_time)
        end_index = None if end_time is None else np.searchsorted(timestamps, end_time)
        return start_index, end_index
    
    @staticmethod
    def _cross_correlate(src_signal, dst_signal):
        crosscorrelation = scipy.signal.correlate(
            dst_signal, src_signal, mode='full')
        lag_indexes = scipy.signal.correlation_lags(
            dst_signal.size, src_signal.size, mode='full')
        return crosscorrelation, lag_indexes

    def _correlation_coefficient(self, sampling_offset, src_signal):
        src_overlap, dst_overlap = self._obtain_overlapping_signals(
            sampling_offset, src_signal)
        return np.corrcoef(src_overlap, dst_overlap)[0, 1]

    def _obtain_overlapping_signals(self, sampling_offset, src_signal):
        """Returns resampled source and destination signals clipped to their overlapping region based on a sampling offset. Used to calculate the Pearson correlation coefficient.

        Args:
            sampling_offset (int): The displacement of the source signal with respect to the destination signal
            source_signal (ndarray): Source signal to be clipped. Might not be the full signal.

        Returns:
            tuple: A tuple containing two arrays: 
                - The given source signal clipped to the overlapping region.
                - The destination signal clipped to the overlapping region.
        """
        if sampling_offset < 0:
            src_trimmed = src_signal[abs(sampling_offset):]
            dst_trimmed = self.dst_resampled
        else:
            src_trimmed = src_signal
            dst_trimmed = self.dst_resampled[sampling_offset: ]  

        n_dst = len(dst_trimmed)
        n_src = len(src_trimmed)
        N = min(n_dst, n_src)
        return src_trimmed[:N], dst_trimmed[:N]
        # if len(dst_trimmed) > len(src_trimmed):
        #     dst_trimmed = dst_trimmed[:len(src_trimmed)]
        # else:
        #     src_trimmed = src_trimmed[:len(dst_trimmed)]
        # return src_trimmed, dst_trimmed
