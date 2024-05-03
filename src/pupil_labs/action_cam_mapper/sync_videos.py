import numpy as np
import scipy.signal
import scipy.interpolate
import pandas as pd


class OffsetCalculator():
    def __init__(self, source, source_timestamps, destination, destination_timestamps, resampling_frequency=100):
        self.source = np.asarray(source).reshape(-1)
        self.source_timestamps = np.asarray(source_timestamps).reshape(-1)
        assert len(self.source) == len(
            self.source_timestamps), "Source signal and timestamps must have the same length"
        self.destination = np.asarray(destination).reshape(-1)
        self.destination_timestamps = np.asarray(destination_timestamps).reshape(-1)
        assert len(self.destination) == len(
            self.destination_timestamps), "Destination signal and timestamps must have the same length"
        self.resampling_frequency = resampling_frequency
        self.source_resampled, self.source_resampled_timestamps = self._resample_signal(
            self.source, self.source_timestamps)
        self.destination_resampled, self.destination_resampled_timestamps = self._resample_signal(
            self.destination, self.destination_timestamps)
        self.time_offset = None

    def _resample_signal(self, signal, timestamps):
        resampled_timestamps = np.arange(
            timestamps[0], timestamps[-1], 1/self.resampling_frequency)
        resampled_signal = scipy.interpolate.interp1d(
            timestamps, signal, 'linear', bounds_error=False, fill_value='extrapolate')(resampled_timestamps)
        return resampled_signal, resampled_timestamps

    def estimate_time_offset(self, source_start_time=None, source_end_time=None):
        if source_start_time is not None and source_end_time is not None:
            assert (source_start_time <
                    source_end_time), f"Start time ({source_start_time} s) must be smaller than end time ({source_end_time} s)"
        start_index, end_index = self._obtain_indexes(
            self.source_resampled_timestamps, source_start_time, source_end_time)
        crosscorrelation, lag_indexes = self._cross_correlate(
            self.source_resampled[start_index:end_index])
        sampling_offset = lag_indexes[np.argmax(crosscorrelation)]
        correlation_score = self._correlation_coefficient(
            sampling_offset, self.source_resampled[start_index:end_index])
        self.time_offset = (sampling_offset / self.resampling_frequency) - \
            self.source_resampled_timestamps[start_index] + \
            self.destination_resampled_timestamps[0]
        return self.time_offset, correlation_score

    def _obtain_indexes(self, timestamps, start_time, end_time):
        start_index = 0 if start_time is None else np.where(
            timestamps >= start_time)[0][0]
        end_index = None if end_time is None else np.where(
            timestamps <= end_time)[0][-1]
        return start_index, end_index

    def _cross_correlate(self, source_signal):
        crosscorrelation = scipy.signal.correlate(
            self.destination_resampled, source_signal, mode='full')
        lag_indexes = scipy.signal.correlation_lags(
            self.destination_resampled.size, source_signal.size, mode='full')
        return crosscorrelation, lag_indexes

    def _correlation_coefficient(self, sampling_offset, source_signal):
        source_overlap, destination_overlapp = self._obtain_overlapping_signals(
            sampling_offset, source_signal)
        return np.corrcoef(source_overlap, destination_overlapp)[0, 1]

    def _obtain_overlapping_signals(self, sampling_offset, source_signal):
        source_clipped = source_signal[abs(
            sampling_offset):] if sampling_offset < 0 else source_signal
        destination_clipped = self.destination_resampled[sampling_offset:
                                                         ] if sampling_offset > 0 else self.destination_resampled
        if len(destination_clipped) > len(source_clipped):
            destination_clipped = destination_clipped[:len(source_clipped)]
        else:
            source_clipped = source_clipped[:len(destination_clipped)]
        return source_clipped, destination_clipped
