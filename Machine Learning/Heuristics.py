from MLData import *
from tqdm import tqdm
from IPython.display import clear_output
from scipy.stats import kurtosis, skew, iqr
from entropy import spectral_entropy
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def get_labels():
    labels = [
        'mean',
        'variance',
        'std_dev',
        'kurtosis',
        'skewness',
        'min_val',
        'max_val',
        'perc25',
        'perc75',
        'inter_quart_range',
        'auto_corr_seq',
        # Frequency Domain Features
        'freq_1',
        'amp_1',
        'freq_2',
        'amp_2',
        'shannon_spectral_entropy',
    ]

    finalLabels = list()

    axes = ['x', 'y', 'z']
    sensors = ['accel', 'gyro']

    finalLabels = [[l + '_' + a for l in labels] for a in axes]
    finalLabels = list(np.array(finalLabels).flatten())
    finalLabels = np.array([[fl + '_' + s for fl in finalLabels]
                            for s in sensors])

    return finalLabels.flatten()


def HeuristicBuilder(feature):
    # Time Domain Features
    heuristicFeature = np.empty(0)
    feature = feature.flatten()
    #index x,y,z at a time

    freqs, amps = fft_coeff(feature)
    heuristics = {
        'mean':
        np.mean(feature),
        'variance':
        np.var(feature),
        'std_dev':
        np.std(feature),
        'kurtosis':
        kurtosis(feature.flatten()),
        'skewness':
        skew(a=feature.flatten()),
        'min_val':
        feature.min(),
        'max_val':
        feature.max(),
        'perc25':
        np.percentile(feature, 25),
        'perc75':
        np.percentile(feature, 25),
        'inter_quart_range':
        iqr(feature),
        'auto_corr_seq':
        _autocorrelation(feature),

        # Frequency Domain Features
        'freq_1':
        freqs[0],
        'amp_1':
        amps[0],
        'freq_2':
        freqs[0],
        'amp_2':
        amps[1],
        'shannon_spectral_entropy':
        spectral_entropy(feature,
                         sf=50,
                         method='fft',
                         nperseg=25,
                         normalize=True)

        # Time and Frequency domain features
        #DWT norms
    }

    heuristicFeature = np.hstack(list(heuristics.values()))

    return (heuristicFeature)


def _autocorrelation(feature):

    result = np.correlate(feature, feature)
    #result = result[int(len(result) / 2):][:3]
    return (result)


def fft_coeff(feat):
    feat = feat.flatten()
    feat = np.sqrt(np.square(feat))
    feat = feat - np.mean(feat)

    #AMPLITUDE
    signal = abs((np.fft.fft(feat) / len(feat))[:int(len(feat) / 4)])

    #FREQUENCY
    signal1 = np.fft.fftfreq(len(feat), d=1 / 50)[:int(len(feat) / 4)]

    ratio = 0
    max_freq_index = 0
    min_freq_index = 0

    peaks, _ = find_peaks(signal)
    # peaks = np.array([p for p in peaks if (signal1[p] > 0.35) & (signal1[p] < 3.1)])

    higher_class_freqs = signal[peaks].argsort()[-1:]
    lower_class_freqs = signal[peaks].argsort()[-10:-1]

    select_freq_df = pd.DataFrame(columns=['Fh', 'Fl', 'ratio_index'])

    for Fh in higher_class_freqs:
        for Fl in lower_class_freqs:
            if signal1[peaks[Fh]] < signal1[peaks[Fl]]:
                ratio = signal1[peaks[Fl]] / signal1[peaks[Fh]]
                select_freq_df = select_freq_df.append(
                    {
                        'Fh': Fh,
                        'Fl': Fl,
                        'ratio_index': np.square(ratio - 2)
                    },
                    ignore_index=True)
            else:
                ratio = signal1[peaks[Fh]] / signal1[peaks[Fl]]
                select_freq_df = select_freq_df.append(
                    {
                        'Fh': Fh,
                        'Fl': Fl,
                        'ratio_index': np.square(ratio - 2)
                    },
                    ignore_index=True)

    select_freq_df = select_freq_df.sort_values(by='ratio_index')
    select_freq_df.reset_index(inplace=True, drop=True)

    top_two_peaks = list(select_freq_df[['Fh', 'Fl']].loc[0])
    top_two_peaks[0] = int(top_two_peaks[0])
    top_two_peaks[1] = int(top_two_peaks[1])

    if signal1[peaks[top_two_peaks[0]]] < signal1[peaks[top_two_peaks[1]]]:
        ratio = signal1[peaks[top_two_peaks[1]]] / signal1[peaks[
            top_two_peaks[0]]]
        max_freq_index = peaks[top_two_peaks[1]]
        min_freq_index = peaks[top_two_peaks[0]]

    else:
        ratio = signal1[peaks[top_two_peaks[0]]] / signal1[peaks[
            top_two_peaks[1]]]
        max_freq_index = peaks[top_two_peaks[0]]
        min_freq_index = peaks[top_two_peaks[1]]

    freqs = [signal1[max_freq_index], signal1[min_freq_index]]
    amps = [signal[max_freq_index], signal[min_freq_index]]

    return (freqs, amps)
