import numpy as np
import scipy.signal
import scipy.interpolate
import pandas as pd


class OffsetCalculator():
    def __init__(self, src_signal, src_timestamps, dst_signal, dst_timestamps, resampling_frequency=100): 
        self.src_signal = src_signal 
        self.src_timestamps = src_timestamps
        self.dst_signal = dst_signal        
        self.dst_timestamps = dst_timestamps
        self.resampling_frequency = resampling_frequency
        self.resampled_src_signal, self.resampled_src_timestamps = self._resample_signal(self.src_signal, self.src_timestamps) 
        self.resampled_dst_signal, self.resampled_dst_timestamps = self._resample_signal(self.dst_signal)
        self.time_offset = None

    def _resample_signal(self, signal, timestamps):
        resampled_timestamps = np.arange(timestamps[0], timestamps[-1], 1/self.resampling_frequency) 
        resampled_signal = scipy.interpolate.interp1d(timestamps, signal,'linear',bounds_error=False, fill_value='extrapolate')(resampled_timestamps)
        return resampled_signal, resampled_timestamps 
    
    def estimate_time_offset(self, src_start_time=None, src_end_time=None):
        crosscorrelation, lag_indexes = self._cross_correlate_signals(self.resampled_dst_signal, self.resampled_src_signal[src_start_time:src_end_time])
        sampling_offset = lag_indexes[np.argmax(crosscorrelation)] # the offset is the lag that maximizes the cross-correlation in the resampled signals
        correlation_score = self._correlation_coefficient(sampling_offset)
        self.time_offset = (sampling_offset / self.resampling_frequency) - self.resampled_src_timestamps[src_start_time] + self.resampled_dst_timestamps[0] 
        return self.time_offset, correlation_score

    def _cross_correlate_signals(self, src_signal):
        crosscorrelation = scipy.signal.correlate(self.resampled_dst_signal,src_signal, mode='full')
        lag_indexes= scipy.signal.correlation_lags(self.resampled_dst_signal.size,src_signal.size, mode='full')
        return crosscorrelation, lag_indexes

    def _correlation_coefficient(self, sampling_offset): #peason's correlation coefficient
        pass
        # To calculate the correlation coefficient between the two aligned signals
        # the signals must be the same length (only the overlapping part of the signals is used for the calculation)
        # return the correlation coefficient np.corrcoef(dst_signal_samelength,src_signal_samelength)[0,1]

