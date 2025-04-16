"""
#__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
#__version__ = "1.0.4"
#__license__ = "MIT"
@authors: A.Maggi 2016 > Orignial code and porting from Matlab
          C. Hibert after 22/05/2017_1 > Original code from Matlab and addition of spectrogram attributes and other stuffs + comments
This function computes the attributes of a seismic signal later used to perform identification through machine
learning algorithms.
- Example: from ComputeAttributes_CH_V1 import calculate_all_attributes
           all_attributes = calculate_all_attributes(Data,sps,flag)
- Inputs: "Data" is the raw seismic signal of the event (cutted at the onset and at the end of the signal)
          "sps" is the sampling rate of the seismic signal (in samples per second)
          "flag" is used to indicate if the input signal is 3C (flag==1) or 1C (flag==0).
          /!\ 3C PROCESSING NOT FULLY IMPLEMENTED YET /!\
- Output: "all_attributes" is an array of the attribute values for the input signal, ordered as detailed on lines 69-137
"""

# <editor-fold desc="**0** detect_peaks.py">


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

#Detect peaks in data-60s based on their amplitude and other features.
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data-60s based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data-60s.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data-60s).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data-60s in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data-60s: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
plt.show()

# </editor-fold>




# <editor-fold desc="**1** ComputeAttributesV2.py">
import numpy as np
from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr
#from detect_peak_modify.py import detect_peaks

def calculate_all_attributes(Data, sps, flag):
    # for 3C make sure is in right order (Z then horizontals)
    if flag==1:
        NATT = 62
    if flag==0:
        NATT = 58 + 2 # (add RMS and iq)

    all_attributes = np.empty((1, NATT), dtype=float)

    env = envelope(Data,sps)

    TesMEAN, TesMEDIAN, TesSTD, env_max = get_TesStuff(env)

    RappMaxMean, RappMaxMedian = get_RappMaxStuff(TesMEAN, TesMEDIAN)

    AsDec, DistDecAmpEnv = get_AsDec(Data, env, sps)

    KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig = get_KurtoSkewStuff(Data, env)

    CorPeakNumber, INT1, INT2, INT_RATIO = get_CorrStuff(Data, sps)

    ES, KurtoF = get_freq_band_stuff(Data, sps)

    MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1, Fquart3,\
    NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1, gamma2,\
    gammas = get_full_spectrum_stuff(Data, sps)

    if flag==1: #If signal is 3C then compute polarisation parameter
        rectilinP, azimuthP, dipP, Plani = get_polarization_stuff(Data, env)

    SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1 \
    = get_pseudo_spectral_stuff(Data, sps)

    # waveform
    all_attributes[0, 0] = np.mean(duration(Data,sps))  # 1  Duration of the signal
    all_attributes[0, 1] = np.mean(RappMaxMean)         # 2  Ratio of the Max and the Mean of the normalized envelope
    all_attributes[0, 2] = np.mean(RappMaxMedian)       # 3  Ratio of the Max and the Median of the normalized envelope
    all_attributes[0, 3] = np.mean(AsDec)               # 4  Ascending time/Decreasing time of the envelope
    all_attributes[0, 4] = np.mean(KurtoSig)            # 5  Kurtosis Signal
    all_attributes[0, 5] = np.mean(KurtoEnv)            # 6  Kurtosis Envelope
    all_attributes[0, 6] = np.mean(np.abs(SkewnessSig)) # 7  Skewness Signal
    all_attributes[0, 7] = np.mean(np.abs(SkewnessEnv)) # 8  Skewness envelope
    all_attributes[0, 8] = np.mean(CorPeakNumber)       # 9  Number of peaks in the autocorrelation function
    all_attributes[0, 9] = np.mean(INT1)                #10  Energy in the 1/3 around the origin of the autocorr function
    all_attributes[0, 10] = np.mean(INT2)               #11  Energy in the last 2/3 of the autocorr function
    all_attributes[0, 11] = np.mean(INT_RATIO)          #12  Ratio of the energies above
    all_attributes[0, 12] = np.mean(ES[0])              #13  Energy of the seismic signal in the 1-3Hz FBand
    all_attributes[0, 13] = np.mean(ES[1])              #14  Energy of the seismic signal in the 2-6Hz FBand
    all_attributes[0, 14] = np.mean(ES[2])              #15  Energy of the seismic signal in the 3-9Hz FBand
    all_attributes[0, 15] = np.mean(ES[3])              #16  Energy of the seismic signal in the 1-5Hz FBand
    all_attributes[0, 16] = np.mean(ES[4])              #17  Energy of the seismic signal in the 4-12Hz FFBand
    all_attributes[0, 17] = np.mean(KurtoF[0])          #18  Kurtosis of the signal in the 1-3Hz FBand
    all_attributes[0, 18] = np.mean(KurtoF[1])          #19  Kurtosis of the signal in the 2-6Hz FBand
    all_attributes[0, 19] = np.mean(KurtoF[2])          #20  Kurtosis of the signal in the 3-9Hz FBand
    all_attributes[0, 20] = np.mean(KurtoF[3])          #21  Kurtosis of the signal in the 1-5Hz FBand
    all_attributes[0, 21] = np.mean(KurtoF[4])          #22  Kurtosis of the signal in the 4-12Hz FBand
    all_attributes[0, 22] = np.mean(DistDecAmpEnv)      #23  Difference bewteen decreasing coda amplitude and straight line
    all_attributes[0, 23] = np.mean(env_max/duration(Data,sps)) # 24  Ratio between max envelope and duration
    # new features from M. Chmiel (ETH Zürich) and added by Qi Zhou, the following #ID were updated
    all_attributes[0, 24] = np.sqrt(np.mean(Data ** 2)) #25  Root mean square
    all_attributes[0, 25] = iqr(np.abs(Data))                   #26  Interquartile range Q75-Q25

    # spectral
    all_attributes[0, 26] = np.mean(MeanFFT)            #27  Mean FFT
    all_attributes[0, 27] = np.mean(MaxFFT)             #28  Max FFT
    all_attributes[0, 28] = np.mean(FmaxFFT)            #29  Frequence at Max(FFT)
    all_attributes[0, 29] = np.mean(FCentroid)          #30  Fq of spectrum centroid
    all_attributes[0, 30] = np.mean(Fquart1)            #31  Fq of 1st quartile
    all_attributes[0, 31] = np.mean(Fquart3)            #32  Fq of 3rd quartile
    all_attributes[0, 32] = np.mean(MedianFFT)          #33  Median Normalized FFT spectrum
    all_attributes[0, 33] = np.mean(VarFFT)             #34  Var Normalized FFT spectrum
    all_attributes[0, 34] = np.mean(NpeakFFT)           #35  Number of peaks in normalized FFT spectrum
    all_attributes[0, 35] = np.mean(MeanPeaksFFT)       #36  Mean peaks value for peaks>0.7
    all_attributes[0, 36] = np.mean(E1FFT)              #37  Energy in the 1 -- NyF/4 Hz (NyF=Nyqusit Freq.) band
    all_attributes[0, 37] = np.mean(E2FFT)              #38  Energy in the NyF/4 -- NyF/2 Hz band
    all_attributes[0, 38] = np.mean(E3FFT)              #39  Energy in the NyF/2 -- 3*NyF/4 Hz band
    all_attributes[0, 39] = np.mean(E4FFT)              #40  Energy in the 3*NyF/4 -- NyF/2 Hz band
    all_attributes[0, 40] = np.mean(gamma1)             #41  Spectrim centroid
    all_attributes[0, 41] = np.mean(gamma2)             #42  Spectrim gyration radio
    all_attributes[0, 42] = np.mean(gammas)             #43  Spectrim centroid width

    # Pseudo-Spectro.
    all_attributes[0, 43] = np.mean(SpecKurtoMaxEnv)    #44  Kurto of the envelope of the maximum energy on spectros
    all_attributes[0, 44] = np.mean(SpecKurtoMedianEnv) #45  Kurto of the envelope of the median energy on spectros
    all_attributes[0, 45] = np.mean(RATIOENVSPECMAXMEAN)#46  Ratio Max DFT(t)/ Mean DFT(t)
    all_attributes[0, 46] = np.mean(RATIOENVSPECMAXMEDIAN)#47  Ratio Max DFT(t)/ Median DFT(t)
    all_attributes[0, 47] = np.mean(DISTMAXMEAN)        #48  Nbr peaks Max DFTs(t)
    all_attributes[0, 48] = np.mean(DISTMAXMEDIAN)      #49  Nbr peaks Mean DFTs(t)
    all_attributes[0, 49] = np.mean(NBRPEAKMAX)         #50  Nbr peaks Median DFTs(t)
    all_attributes[0, 50] = np.mean(NBRPEAKMEAN)        #51  Ratio Max/Mean DFTs(t)
    all_attributes[0, 51] = np.mean(NBRPEAKMEDIAN)      #52  Ratio Max/Median DFTs(t)
    all_attributes[0, 52] = np.mean(RATIONBRPEAKMAXMEAN)#53  Nbr peaks X centroid Freq DFTs(t)
    all_attributes[0, 53] = np.mean(RATIONBRPEAKMAXMED) #54  Nbr peaks X Max Freq DFTs(t)
    all_attributes[0, 54] = np.mean(NBRPEAKFREQCENTER)  #55  Ratio Freq Max/X Centroid DFTs(t)
    all_attributes[0, 55] = np.mean(NBRPEAKFREQMAX)     #56  Mean distance bewteen Max DFT(t) Mean DFT(t)
    all_attributes[0, 56] = np.mean(RATIONBRFREQPEAKS)  #57  Mean distance bewteen Max DFT Median DFT
    all_attributes[0, 57] = np.mean(DISTQ2Q1)           #58  Distance Q2 curve to Q1 curve (QX curve = envelope of X quartile of DTFs)
    all_attributes[0, 58] = np.mean(DISTQ3Q2)           #59  Distance Q3 curve to Q2 curve
    all_attributes[0, 59] = np.mean(DISTQ3Q1)           #60  Distance Q3 curve to Q1 curve

    # polarisation
    if flag==1:
        all_attributes[0, 60] = rectilinP
        all_attributes[0, 61] = azimuthP
        all_attributes[0, 62] = dipP
        all_attributes[0, 63] = Plani

    return all_attributes


# -----------------------------------#
#        Secondary Functions         #
# -----------------------------------#

def duration(Data,sps):
    dur = len(Data) / sps
    return dur


def envelope(Data,sps):
    env = np.abs(hilbert(Data))
    return env


def get_TesStuff(env):
    CoefSmooth=3
    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)


    env_max = np.max(env)
    tmp = lfilter(light_filter, 1, env/env_max)
    TesMEAN = np.mean(tmp)
    TesMEDIAN = np.median(tmp)
    TesSTD = np.std(tmp)

    return TesMEAN, TesMEDIAN, TesSTD, env_max


def get_RappMaxStuff(TesMEAN, TesMEDIAN):
    npts = 1
    RappMaxMean = np.empty(npts, dtype=float)
    RappMaxMedian = np.empty(npts, dtype=float)

    #for i in range(npts):
    RappMaxMean = 1./TesMEAN
    RappMaxMedian = 1./TesMEDIAN

    return RappMaxMean, RappMaxMedian


def get_AsDec(Data, env, sps):
    strong_filter = np.ones(int(sps)) / float(sps)

    smooth_env = lfilter(strong_filter, 1, env)
    imax = np.argmax(smooth_env)

    if float(len(Data) - (imax+1))>0:
        AsDec = (imax+1) / float(len(Data) - (imax+1))
    else:
        AsDec = 0 

    dec = Data[imax:]
    lendec = len(dec)

    DistDecAmpEnv = np.abs(np.mean(np.abs(hilbert(dec / np.max(Data))) -
            (1 - ((1 / float(lendec)) * (np.arange(lendec)+1)))))

    return AsDec, DistDecAmpEnv


def get_KurtoSkewStuff(Data, env):
    ntr = 1

    KurtoEnv = np.empty(ntr, dtype=float)
    KurtoSig = np.empty(ntr, dtype=float)
    SkewnessEnv = np.empty(ntr, dtype=float)
    SkewnessSig = np.empty(ntr, dtype=float)
    CoefSmooth = 3

    light_filter = np.ones(CoefSmooth) / float(CoefSmooth)

    env_max = np.max(env)
    data_max = np.max(Data)
    tmp = lfilter(light_filter, 1, env/env_max)
    KurtoEnv = kurtosis(tmp, fisher=False)
    SkewnessEnv = skew(tmp)
    KurtoSig = kurtosis(Data / data_max, fisher=False)
    SkewnessSig = skew(Data / data_max)

    return KurtoEnv, KurtoSig, SkewnessEnv, SkewnessSig


def get_CorrStuff(Data,sps):
    strong_filter = np.ones(int(sps)) / float(sps)
    min_peak_height = 0.4

    ntr=1
    CorPeakNumber = np.empty(ntr, dtype=int)
    INT1 = np.empty(ntr, dtype=float)
    INT2 = np.empty(ntr, dtype=float)
    INT_RATIO = np.empty(ntr, dtype=float)

    cor = np.correlate(Data, Data, mode='full')
    cor = cor / np.max(cor)

    # find number of peaks
    cor_env = np.abs(hilbert(cor))
    cor_smooth = lfilter(strong_filter, 1, cor_env)
    cor_smooth2 = lfilter(strong_filter, 1, cor_smooth/np.max(cor_smooth))
    ipeaks = detect_peaks(cor_smooth2,min_peak_height)

    CorPeakNumber =len(ipeaks)

    # integrate over bands
    npts = len(cor_smooth)
    ilag_0 = np.argmax(cor_smooth)+1
    ilag_third = round(ilag_0 + npts/6) #ilag_0 + npts/6 for # 720045.1666666666 # TypeError: slice indices must be integers or None or have an __index__ method


    max_cor = np.max(cor_smooth)
    int1 = np.trapz(cor_smooth[ilag_0:ilag_third+1]/max_cor)
    int2 = np.trapz(cor_smooth[ilag_third:]/max_cor)
    int_ratio = int1 / int2

    INT1 = int1
    INT2 = int2
    INT_RATIO = int_ratio

    return CorPeakNumber, INT1, INT2, INT_RATIO


def get_freq_band_stuff(Data, sps):
    NyF = sps / 2

    if sps <= 50:
        FFI = np.array([1, 5,  15, 15, 15]) # lower bounds of the different tested freq. bands
        FFE = np.array([5, 15, 24, 24, 24])  # higher bounds of the different tested freq. bands

    else:
        FFI = np.array([1, 5,  15, 25, 35]) # lower bounds of the different tested freq. bands
        FFE = np.array([5, 15, 25, 35, 45])  # higher bounds of the different tested freq. bands

    nf = len(FFI)

    ES = np.empty(nf, dtype=float)
    KurtoF = np.empty(nf, dtype=float)

    for j in range(nf):
        Fb, Fa = butter(2, [FFI[j] / NyF, FFE[j] / NyF], 'bandpass')
        data_filt = lfilter(Fb, Fa, Data)

        ES[j] = np.log10(np.trapz(np.abs(hilbert(data_filt))))
        KurtoF[j] = kurtosis(data_filt, fisher=False)

    return ES, KurtoF


def get_full_spectrum_stuff(Data,sps):
    NyF = sps / 2

    ntr = 1
    MeanFFT = np.empty(ntr, dtype=float)
    MaxFFT = np.empty(ntr, dtype=float)
    FmaxFFT = np.empty(ntr, dtype=float)
    MedianFFT = np.empty(ntr, dtype=float)
    VarFFT = np.empty(ntr, dtype=float)
    FCentroid = np.empty(ntr, dtype=float)
    Fquart1 = np.empty(ntr, dtype=float)
    Fquart3 = np.empty(ntr, dtype=float)
    NpeakFFT = np.empty(ntr, dtype=float)
    MeanPeaksFFT = np.empty(ntr, dtype=float)
    E1FFT = np.empty(ntr, dtype=float)
    E2FFT = np.empty(ntr, dtype=float)
    E3FFT = np.empty(ntr, dtype=float)
    E4FFT = np.empty(ntr, dtype=float)
    gamma1 = np.empty(ntr, dtype=float)
    gamma2 = np.empty(ntr, dtype=float)
    gammas = np.empty(ntr, dtype=float)

    b = np.ones(300) / 300.0

    data = Data
    npts = 2560
    n = nextpow2(2*npts-1)
    Freq1 = np.linspace(0, 1, int(n/2) ) * NyF # TypeError: 'float' object cannot be interpreted as an integer, add int()

    FFTdata = 2 * np.abs(np.fft.fft(data, n=n)) / float(npts * npts)
    FFTsmooth = lfilter( b, 1, FFTdata[ 0 : int(len(FFTdata)/2) ] ) #TypeError: slice indices must be integers or None or have an __index__ method, add int()
    FFTsmooth_norm = FFTsmooth / max(FFTsmooth)

    MeanFFT = np.mean(FFTsmooth_norm)
    MedianFFT = np.median(FFTsmooth_norm)
    VarFFT = np.var(FFTsmooth_norm, ddof=1)
    MaxFFT = np.max(FFTsmooth)
    iMaxFFT = np.argmax(FFTsmooth)
    FmaxFFT = Freq1[iMaxFFT]

    xCenterFFT = np.sum((np.arange(len(FFTsmooth_norm))) * FFTsmooth_norm) / np.sum(FFTsmooth_norm)
    i_xCenterFFT = int(np.round(xCenterFFT))

    xCenterFFT_1quart = np.sum((np.arange(i_xCenterFFT+1)) * FFTsmooth_norm[0:i_xCenterFFT+1]) / np.sum(FFTsmooth_norm[0:i_xCenterFFT+1])

    i_xCenterFFT_1quart = int(np.round(xCenterFFT_1quart))

    xCenterFFT_3quart = np.sum((np.arange(len(FFTsmooth_norm) - i_xCenterFFT)) * FFTsmooth_norm[i_xCenterFFT:]) / \
                        np.sum(FFTsmooth_norm[i_xCenterFFT:]) + i_xCenterFFT+1

    i_xCenterFFT_3quart = int(np.round(xCenterFFT_3quart))

    FCentroid = Freq1[i_xCenterFFT]
    Fquart1 = Freq1[i_xCenterFFT_1quart]
    Fquart3 = Freq1[i_xCenterFFT_3quart]

    min_peak_height = 0.75
    ipeaks = detect_peaks(FFTsmooth_norm,min_peak_height,100)

    NpeakFFT = len(ipeaks)
    sum_peaks=0

    for ll in range(0,len(ipeaks)):
        sum_peaks+=FFTsmooth_norm[ipeaks[ll]]

    if NpeakFFT>0:
        MeanPeaksFFT = sum_peaks / float(NpeakFFT)
    else:
        MeanPeaksFFT= 0

    npts = len(FFTsmooth_norm)

    E1FFT = np.trapz(FFTsmooth_norm[0:round(npts/4)])# replace the npts/4 with round(npts/4)
    E2FFT = np.trapz(FFTsmooth_norm[round(npts/4)-1:2*round(npts/4)])
    E3FFT = np.trapz(FFTsmooth_norm[2*round(npts/4)-1:3*round(npts/4)])
    E4FFT = np.trapz(FFTsmooth_norm[3*round(npts/4)-1:npts])

    moment = np.empty(3, dtype=float)

    for j in range(3):
        moment[j] = np.sum(Freq1**j * FFTsmooth_norm[0:int(n/2)]**2)

    gamma1 = moment[1]/moment[0]
    gamma2 = np.sqrt(moment[2]/moment[0])
    gammas = np.sqrt(np.abs(gamma1**2 - gamma2**2))

    return MeanFFT, MaxFFT, FmaxFFT, MedianFFT, VarFFT, FCentroid, Fquart1,\
        Fquart3, NpeakFFT, MeanPeaksFFT, E1FFT, E2FFT, E3FFT, E4FFT, gamma1,\
        gamma2, gammas


def get_polarization_stuff(st, env):
    sps = st[0].stats.sampling_rate
    strong_filter = np.ones(int(sps)) / float(sps)
    smooth_env = lfilter(strong_filter, 1, env[0])
    imax = np.argmax(smooth_env)
    end_window = int(np.round(imax/3.))

    xP = st[2].data[0:end_window]
    yP = st[1].data[0:end_window]
    zP = st[0].data[0:end_window]

    MP = np.cov(np.array([xP, yP, zP]))
    w, v = np.linalg.eig(MP)

    indexes = np.argsort(w)
    DP = w[indexes]
    pP = v[:, indexes]

    rectilinP = 1 - ((DP[0] + DP[1]) / (2*DP[2]))
    azimuthP = np.arctan(pP[1, 2] / pP[0, 2]) * 180./np.pi
    dipP = np.arctan(pP[2, 2] / np.sqrt(pP[1, 2]**2 + pP[0, 2]**2)) * 180/np.pi
    Plani = 1 - (2 * DP[0]) / (DP[1] + DP[2])

    return rectilinP, azimuthP, dipP, Plani


def get_pseudo_spectral_stuff(Data, sps):
    ntr=1
    SpecKurtoMaxEnv = np.empty(ntr, dtype=float)
    SpecKurtoMedianEnv = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEAN = np.empty(ntr, dtype=float)
    RATIOENVSPECMAXMEDIAN = np.empty(ntr, dtype=float)
    DISTMAXMEAN = np.empty(ntr, dtype=float)
    DISTMAXMEDIAN = np.empty(ntr, dtype=float)
    NBRPEAKMAX = np.empty(ntr, dtype=float)
    NBRPEAKMEAN  = np.empty(ntr, dtype=float)
    NBRPEAKMEDIAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMEAN = np.empty(ntr, dtype=float)
    RATIONBRPEAKMAXMED = np.empty(ntr, dtype=float)
    NBRPEAKFREQCENTER = np.empty(ntr, dtype=float)
    NBRPEAKFREQMAX = np.empty(ntr, dtype=float)
    RATIONBRFREQPEAKS = np.empty(ntr, dtype=float)
    DISTQ2Q1 = np.empty(ntr, dtype=float)
    DISTQ3Q2 = np.empty(ntr, dtype=float)
    DISTQ3Q1 = np.empty(ntr, dtype=float)

    # Spectrogram parametrisation
    SpecWindow = 100 # Window legnth
    noverlap = int(0.90 * SpecWindow) # Overlap
    n = 2048 
    Freq=np.linspace(0, sps, int(n/2) ) # Sampling of frequency array
    b_filt = np.ones(100) / 100.0 # Smoothing param

    # Spectrogram computation from DFT (Discrete Fourier Transform on a moving window)
    f, t, spec = spectrogram(Data, fs=sps, window='boxcar',
                                     nperseg=SpecWindow, nfft=n, noverlap=noverlap,
                                     scaling='spectrum')

    smooth_spec = lfilter(b_filt, 1, np.abs(spec), axis=1) #smoothing

    # Envelope of the maximum of each DFT constituting the spectrogram
    SpecMaxEnv,SpecMaxFreq = smooth_spec[0:800,:].max(0),smooth_spec[0:800,:].argmax(0)

    # Envelope of the mean of each DFT constituting the spectrogram
    SpecMeanEnv=smooth_spec.mean(0)

    # Envelope of the median of each DFT constituting the spectrogram
    SpecMedianEnv=np.median(smooth_spec,0)

    # Envelope of different quartiles of each DFT
    CentoiX=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX1=np.empty(np.size(smooth_spec,1), dtype=float)
    CentoiX3=np.empty(np.size(smooth_spec,1), dtype=float)

    # Envelope of the frequencies corresponding to different quartiles of the DFT
    for v in range(0, np.size(smooth_spec,1) ):
        CentroIndex=np.around(centeroidnpX(smooth_spec[0:800,v]))
        CentroIndex = int(CentroIndex)
        CentoiX[v] = (Freq[ CentroIndex ] ) # add int()
        CentoiX1[v] = ( Freq[ int(np.around(centeroidnpX(smooth_spec[0:CentroIndex,v]))) ] )
        CentoiX3[v] = ( Freq[ int(np.around(centeroidnpX(smooth_spec[CentroIndex:800,v])+CentroIndex)) ] )

    # Tranform into single values
    SpecKurtoMaxEnv=kurtosis(SpecMaxEnv / SpecMaxEnv.max(axis=0))
    SpecKurtoMedianEnv=kurtosis(SpecMedianEnv / SpecMedianEnv.max(axis=0))
    RATIOENVSPECMAXMEAN = np.mean(SpecMaxEnv / SpecMeanEnv)
    RATIOENVSPECMAXMEDIAN = np.mean(SpecMaxEnv / SpecMedianEnv)
    DISTMAXMEAN = np.mean(np.abs(SpecMaxEnv - SpecMeanEnv))
    DISTMAXMEDIAN = np.mean(np.abs(SpecMaxEnv - SpecMedianEnv))
    NBRPEAKMAX = len(detect_peaks(SpecMaxEnv / SpecMaxEnv.max(axis=0),0.75))
    NBRPEAKMEAN  = len(detect_peaks(SpecMeanEnv / SpecMeanEnv.max(axis=0),0.75))
    NBRPEAKMEDIAN = len(detect_peaks(SpecMedianEnv / SpecMedianEnv.max(axis=0),0.75))

    if NBRPEAKMEAN>0:
        RATIONBRPEAKMAXMEAN = np.divide(NBRPEAKMAX, NBRPEAKMEAN) 
    else:
        RATIONBRPEAKMAXMEAN=0

    if NBRPEAKMEDIAN>0:
        RATIONBRPEAKMAXMED = np.divide(NBRPEAKMAX, NBRPEAKMEDIAN)
    else:
        RATIONBRPEAKMAXMED=0

    NBRPEAKFREQCENTER = len(detect_peaks(CentoiX / CentoiX.max(axis=0),0.75))
    NBRPEAKFREQMAX = len(detect_peaks(SpecMaxFreq / SpecMaxFreq.max(axis=0),0.75))

    if NBRPEAKFREQCENTER>0:
        RATIONBRFREQPEAKS = NBRPEAKFREQMAX / NBRPEAKFREQCENTER
    else:
        RATIONBRFREQPEAKS=0

    DISTQ2Q1 = np.mean(abs(CentoiX-CentoiX1))
    DISTQ3Q2 = np.mean(abs(CentoiX3-CentoiX))
    DISTQ3Q1 = np.mean(abs(CentoiX3-CentoiX1))

    return SpecKurtoMaxEnv, SpecKurtoMedianEnv, RATIOENVSPECMAXMEAN, RATIOENVSPECMAXMEDIAN, \
    DISTMAXMEAN , DISTMAXMEDIAN, NBRPEAKMAX, NBRPEAKMEAN, NBRPEAKMEDIAN, RATIONBRPEAKMAXMEAN, \
    RATIONBRPEAKMAXMED, NBRPEAKFREQCENTER, NBRPEAKFREQMAX, RATIONBRFREQPEAKS, DISTQ2Q1, DISTQ3Q2, DISTQ3Q1


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def l2filter(b, a, x):
    # explicit two-pass filtering with no bells or whistles
    x_01 = lfilter(b, a, x)
    x_02 = lfilter(b, a, x_01[::-1])
    x_02 = x_02[::-1]

def centeroidnpX(arr):
    length = np.arange(1,len(arr)+1)
    CentrX=np.sum(length*arr)/np.sum(arr)
    return CentrX

# </editor-fold>
