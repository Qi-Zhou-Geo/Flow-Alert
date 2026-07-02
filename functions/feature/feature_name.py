#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-30
# __author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


def load_feature_name(print_log=False):
    
    # check the paper for details of A features: https://doi.org/10.1029/2024JF007691
    feature_Name_A = ['time_window_start', 'time_stamps', 'station', 'component',
                    'digit1','digit2','digit3','digit4','digit5', 'digit6','digit7','digit8','digit9',
                    'max', 'goodness', 'iqr', 'magnitude_range', 'alpha', 'ks', 'MannWhitneU', 'follow'] # Benford's Law features

    # check the paper for details of B features: https://doi.org/10.1029/2020gl090874, https://doi.org/10.1002/2016gl070709
    feature_Name_B = ['time_window_start', 'time_stamps', 'station', 'component',
                    'RappMaxMean', 'RappMaxMedian', 'AsDec', 'KurtoSig','KurtoEnv', 'SkewnessSig','SkewnessEnv',
                    'CorPeakNumber', 'INT1', 'INT2', 'INT_RATIO', 'ES_0', 'ES_1', 'ES_2', 'ES_3', 'ES_4', 'KurtoF_0',
                    'KurtoF_1', 'KurtoF_2', 'KurtoF_3', 'KurtoF_4', 'DistDecAmpEnv','env_max_to_duration', 'RMS', 'IQR', 'MeanFFT', 'MaxFFT',
                    'FmaxFFT', 'FCentroid', 'Fquart1', 'Fquart3', 'MedianFFT', 'VarFFT', 'NpeakFFT', 'MeanPeaksFFT', 'E1FFT','E2FFT','E3FFT', 'E4FFT',
                    'gamma1', 'gamma2', 'gammas', 'SpecKurtoMaxEnv', 'SpecKurtoMedianEnv', 'RatioEnvSpecMaxMean', 'RatioEnvSpecMaxMedian','DistMaxMean',
                    'DistMaxMedian', 'NbrPeakMax', 'NbrPeakMean', 'NbrPeakMedian','RatioNbrPeakMaxMean', 'RatioNbrPeakMaxMedian',
                    'NbrPeakFreqCenter', 'NbrPeakFreqMax', 'RatioNbrFreqPeaks','DistQ2Q1', 'DistQ3Q2', 'DistQ3Q1'] # Waveform, Spectral, and Spectrogram features

    if print_log is True:
        print(f"load_feature_name:")
        print(f"len(feature_Name_A): {len(feature_Name_A)}")
        print(f"len(feature_Name_B): {len(feature_Name_B)}\n")
    
    return feature_Name_A, feature_Name_B

