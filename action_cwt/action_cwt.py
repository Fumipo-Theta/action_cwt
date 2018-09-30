import pycwt as wavelet
from pycwt.helpers import find
import numpy as np
from func_helper import pip, mapping, filtering, reducing, identity
from matpos import MatPos
import matplotlib.pyplot as plt

from typing import List, Tuple, TypeVar, Union, Callable, Dict, Any, NewType, Type, Optional

"""
[PyCWT: spectral analysis using wavelets in Python — PyCWT 0.3.0a22 documentation]
(https://pycwt.readthedocs.io/en/latest/index.html)

[regeirk/pycwt: A Python module for continuous wavelet spectral analysis.
It includes a collection of routines for wavelet transform and
    statistical analysis via FFT algorithm.
In addition, the module also includes cross-wavelet transforms,
    wavelet coherence tests and sample scripts.](https://github.com/regeirk/pycwt)
"""

Number = Union[int, float, complex]
Limit = Union[List[Number], Tuple[Number, Number]]
T = TypeVar("T")
VectorD = NewType("VectorD", np.ndarray)


def isAllTrue(arr: List[bool]) -> bool:
    return reducing(lambda a, b: a and b)(arr)(True)


def setLimit(lim: Limit) -> Limit:
    return (
        lim[0] if lim[0] != None else - float("inf"),
        lim[1] if lim[1] != None else float("inf")
    )


def isInRange(lim: Limit) -> bool:
    return lambda v: pip(
        setLimit,
        lambda lim: (lim[0] <= v) and (v <= lim[1])
    )(lim)


def isAllIn(obj: Dict[Any, Any]) -> Callable[[List[Any]], bool]:
    return lambda props: isAllTrue(
        mapping(lambda prop: prop in obj)(props)
    )





class CWT:

    def __init__(self, t: VectorD, signal: VectorD, detrend: bool=True, verbose: bool=False) -> None:
        """
        Construct instance for operating and plotting


        Parameters
        ==========
        t: [float] list like ofject
            Time series for signal.
        signal: [float] list like ofject
            Signal.
        detrend: bool, optional
            If true, signal is detrended.
            Default is true.
        verbose: bool, optional
            If true, you can get more information.
            Default is False.

        Example
        =======
        t = numpy.arange(100)
        x = genSignal(t)
        cwt = CWT(t,x)
        """
        self.t = t
        self.signal = signal
        self.N = len(signal)
        self.dat_norm, self.std = pip(
            CWT.detrend if detrend else identity,
            CWT.normalized
        )(signal)
        self.filterPeriod = lambda period: True
        self.verbose = verbose
        self.J = -1
        self.freqs = None
        self.threshold = 0.95

    @staticmethod
    def setup(t: VectorD, signal: VectorD, detrend: bool=True, verbose: bool=False):
        """
        Helper method for initiating method chain style operation.

        Example
        =======
        CWT.setup(t,x)\
            .setTimeInterval(dt)\
            .setMinimumScale(dt*2)\
            .setSubOctave(1/2)\
            .setNumberOfScale()\
            .setMotherWavelet(wavelet.Morlet(6))\
            .auto()\
            .plot()
        """
        return CWT(t, signal, detrend, verbose)

    @staticmethod
    def detrend(x: VectorD) -> VectorD:
        n = len(x)
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)
        return x - np.polyval(p, t)

    @staticmethod
    def normalized(x: VectorD) -> Tuple[VectorD, float]:
        s = x.std()
        return (x/s, s)

    def setScale(
        self,
        dt: Number,
        unit: str="",
        minimum_scale: Optional[List[Number]]=None,
        sub_octave: Number=2,
        number_of_scale: Number=-1,
        freqs: Optional[List[Number]] = None
    ):
        """
        Sugar syntax of setting parameters to define scales.

        Parameters
        ==========
        dt: float
        unit: str, optional
        minimum_scale: float, optional
        sub_octave: float, optional
        number_of_scale: float, optional
        freqs: numpy.ndarray, optional

        Return
        ======
        self

        Example
        =======
        cwt.setScale(dt=1, unit="second", minimum_scale=1*2,
            sub_octave=2
        )
        """

        self.setTimeInterval(dt, unit)\
            .setMinimumScale(minimum_scale)\
            .setSubOctave(sub_octave)\
            .setNumberOfScale(number_of_scale)\
            .setFrequency(freqs)
        return self

    def setTimeInterval(self, dt, unit=""):
        """
        Set real value of time inteval of sampling.
        The analyzed periods and frequencies are relative value
            to the time interval.
        Optionally, you can set unit of time interval.
            It only used in axis label in plotting result.

        Parameters
        ==========
        dt: float
            Time interval of sampling.
        unit: str, optional
            Unit of time (year, minuite, second...)
                used for label of axis in plot.

        Return
        ======
        self

        Example
        =======
        # When sampling interval is 1 minute.
        cwt.setTimeInterval(1,unit="seconds")

        # When sampling interval is 1 month while
        #    you want to show year scale in result:
        cwt.setTimeInterval(1/12, unit="year")
        """

        self.dt = dt
        self.timeUnit = unit
        return self

    def setMinimumScale(self, s0=None):
        """
        Minimum time scale from which the cwt calculated.

        Parameter
        =========
        s0: float, optional
            minimum scale of wavelet.
            Default is dt.

        Return
        ======
        self

        Example
        =======
        # When time interval is 1 year and you want to get
        #  time scale longer than 2 years:
        cwt.setTimeInterval(1,unit="year")
        cwt.setMinimumScale(1 * 2)
        """
        self.s0 = s0 if s0 != None else self.dt * 2
        return self

    def setSubOctave(self, n=2):
        """
        Set number of sub scale in each main time scale.

        Parameter
        =========
        n: float, optional
            sub octave per octaves.
            Default is 2

        Return
        ======
        self

        Example
        =======
        # When time interval is 1 year and you want to analyze
        #  month scale:
        cwt.setTimeInterval(1,unit="year")
        cwt.setSubOctave(12)
        """
        self.dj = 1/n
        return self

    def setNumberOfScale(self, J=-1):
        """
        Set number of scales calculated by cwt.

        Parameter
        =========
        J: float, optional
            Number of scales. Scales range from s0 up to
            s0 * 2**(J * dj), which gives a total of (J + 1) scales.
            Default is J = (log2(N * dt / s0)) / dj.

        Return
        ======
        self

        Example
        =======
        cwt.setNumberOfScale(100)
        """
        if (self.verbose):
            print(
                "Number of scale is",
                np.log2(self.N * self.dt / self.s0)/self.dj if J == -1 else J
            )

        self.J = J
        return self

    def setFrequency(self, freqs=None):
        """
        Set sequence of custom frequency instead of automatically calculated ones
            in pycwt.

        Parameter
        =========
        freqs: numpy.ndarray, optional
            The sequence must be descending order.

        Return
        ======
        self

        Example
        =======
        cwt.setFrequency(numpy.array([64, 32, 16, 8, 4, 2, 1, 0.5, 0.25]))
        """
        self.freqs = freqs
        return self

    def setMothorWavelet(self, mother=wavelet.Morlet(6)):
        """
        Set class of mother wavelet.
        In the pycwt module, general mother wavelet is not implemented
            (for example Morlet wavelet is only for ω0 = 6).

        Parameter
        =========
        mother: obj, optional (Morlet, Paul, Dog, MexicanHat)
            Function of mother wavelet.
            Default is Morlet wavelet with ω0 = 6.

        Return
        ======
        self

        Example
        =======
        import pycwt as wavelet

        cwt.setMotherWavelet(wavelet.Morlet(6))
        cwt.setMotherWavelet(wavelet.Paul())
        """
        self.mother = mother
        return self

    def setPValue(self, p=0.05):
        """
        Set p-value in detecting significant power and scales.

        Parameter
        ========
        p: float, optional
            p-value in the range of [0,1].
            Default is 0.05.

        Return
        ======
        self

        Example
        =======
        cwt.setPValue(0.05)
        """
        if not isInRange([0, 1])(p):
            raise ValueError("p must be in [0,1]")
        self.threshold = 1 - p
        return self

    def filterPeriodBy(self, pred: Callable[[VectorD], bool]=lambda period: True):
        """
        Set predicate function to filter periods for calculating
            average energy.

        Parameter
        =========
        pred: function(pred) -> bool
            Function defining filtering condition for each period.
            The periods matching the condition are used in calculating
                transition of average energy.

        Return
        ======
        self

        Example
        =======
        # For filtering period between unit of 1 and 10:
        cwt.filterPeriodBy(lambda period: (1 <= period) & (period < 10))
        """

        self.filterPeriod = pred
        return self

    def __setupCompleted(self):
        return isAllIn(self)(["dt", "dj", "s0", "mothor"])

    def cwt(self):
        """
        Operating cwt.

        Parameter
        ---------
        None

        Return
        ------
        wave : numpy.ndarray
            Wavelet transform according to the selected mother wavelet.
            Has (J+1) x N dimensions.
        scales : numpy.ndarray
            Vector of scale indices given by sj = s0 * 2**(j * dj),
            j={0, 1, ..., J}.
        freqs : array like
            Vector of Fourier frequencies (in 1 / time units) that
            corresponds to the wavelet scales.
        coi : numpy.ndarray
            Returns the cone of influence, which is a vector of N
            points containing the maximum Fourier period of useful
            information at that particular time. Periods greater than
            those are subject to edge effects.
        fft : numpy.ndarray
            Normalized fast Fourier transform of the input signal.
        fft_freqs : numpy.ndarray
            Fourier frequencies (in 1/time units) for the calculated
            FFT spectrum.

        Example
        -------
        wave, scales, coi, fft, fft_freqs = cwt.cwt()
        """
        self.wave, \
            self.scales, \
            self.freqs, \
            self.coi, \
            self.fft, \
            self.fft_freqs \
            = wavelet.cwt(
                self.dat_norm,
                self.dt,
                self.dj,
                self.s0,
                self.J,
                self.mother,
                self.freqs
            )

        return (
            self.wave,
            self.scales,
            self.freqs,
            self.coi,
            self.fft,
            self.fft_freqs,
        )

    def icwt(self):
        """
        Operate inverse cwt with waves obtained by cwt.

        Parameter
        ---------
        None

        Return
        ------
        iwave : numpy.ndarray
            Inverse wavelet transform.

        Example
        -------
        iwave = cwt.icwt()
        """
        self.iwave = wavelet.icwt(
            self.wave,
            self.scales,
            self.dt,
            self.dj,
            self.mother
        ) * self.std
        return self.iwave

    def getPower(self, rectify=False):
        """
        Calculate the normalized wavelet and Fourier power spectra.
        Optionally, you can also rectify the power spectrum
            according to the suggestions proposed by Liu et al. (2007)[2]

        Parameter
        ---------
        rectify: bool, optional
            Flag for rectification of power.
            Default is False

        Return
        ------
        power: numpy.ndarray
            Power of wavelets.
            Has (J+1) x N dimensions.

        fft_power: numpy.ndarray
            Power of FFT.

        Example
        -------
        power, fft_power = cwt.getPower()
        """
        self.power = (np.abs(self.wave)) ** 2 if rectify == False\
            else (np.abs(self.wave)) ** 2 / self.scales[:, None]
        self.fft_power = np.abs(self.fft) ** 2

        return (self.power, self.fft_power)

    def getPeriod(self):
        """
        Periods of wavelets.

        Parameter
        ---------
        None

        Return
        ------
        period: array like
            Vector of periods in unit of time.

        Example
        -------
        period = cwt.getPeriod()
        """
        self.period = 1 / self.freqs
        return self.period

    def getAlpha(self):
        """
        Lag-1 autocorrelation for red noise
        """
        self.alpha, var, mu2 = wavelet.ar1(self.signal)

        if self.verbose:
            print("Variance of red noise: ", var * self.std**2)
            #print("Mean square of red noise: ", mu2 * self.std**2)

        return self.alpha

    def getSignificantPower(self):
        """
        We could stop at this point and plot our results. However we are also
        interested in the power spectra significance test.
        The power is significant where the ratio power / sig95 > 1.

        Return
        -------
        significantPower : array like
            Significance levels as a function of scale.
        fft_theor (array like):
            Theoretical red-noise spectrum as a function of period.
        """

        signif, self.fft_theor = wavelet.significance(
            1.0,
            self.dt,
            self.scales,
            0,
            self.alpha,
            significance_level=self.threshold,
            wavelet=self.mother
        )

        sig95 = np.ones([1, self.N]) * signif[:, None]

        self.significantPower = self.power / sig95  # sig95
        return (self.significantPower, self.fft_theor)

    def getGlobalSignificance(self):
        """
        Then, we determine significance level
        of the global wavelet spectrum.
        """

        var = self.std**2
        dof = self.N - self.scales  # Correction for padding at edges

        self.glbl_signif, _ = wavelet.significance(
            var,
            self.dt,
            self.scales,
            0.,
            self.alpha,
            significance_level=self.threshold, dof=dof,
            wavelet=self.mother
        )
        return self.glbl_signif

    def getGlobalPower(self):
        """
        各周期のpowerの, 時間方向の平均値.
        """
        self.glbl_power = self.power.mean(axis=1)
        return self.glbl_power

    def getAverageScale(self):
        E = self.std**2
        sel = find(self.filterPeriod(self.period))
        interval = self.dj * self.dt
        Cdelta = self.mother.cdelta

        scale_avg = pip(
            lambda scale: scale.transpose(),
            lambda scale: self.power / scale,
            lambda scaled: scaled[sel, :].sum(axis=0),
            lambda glbl_scale: E * interval * glbl_scale / Cdelta
        )(self.scales * np.ones((self.N, 1)))

        self.scale_avg = scale_avg
        return self.scale_avg

    def getAverageScaleSignificance(self):
        """
        Performs a scale-average test (equations 25 to 28).
        In this case dof should be set to a two element vector [s1, s2],
            which gives the scale range that were averaged together.
        If, for example, the average between scales 2 and 8 was taken, then dof=[2, 8].
        """

        var = self.std**2
        sel = find(self.filterPeriod(self.period))
        scale_avg_signif, _ = wavelet.significance(
            var,
            self.dt,
            self.scales,
            2,
            self.alpha,
            significance_level=self.threshold,
            dof=[
                self.scales[sel[0]], self.scales[sel[-1]]
            ],
            wavelet=self.mother
        )
        self.scale_avg_signif = scale_avg_signif
        return self.scale_avg_signif

    def auto(self):
        """
        Automatic operation of cwt and plotting result.
        """
        if (not self.__setupCompleted):
            raise SystemError("setup not completed.")

        self.cwt()
        self.icwt()
        self.getPower()
        self.getPeriod()
        self.getAlpha()
        self.getSignificantPower()
        self.getGlobalSignificance()
        self.getGlobalPower()
        self.getAverageScale()
        self.getAverageScaleSignificance()

        if (self.verbose):
            print("scale: ", self.scales)
            print("period: ", self.period)
            print("variance of signal:", self.std**2)

        return self

    def plot(self, subgrids=None, style={}):
        """
        Generate figure with 4 sub plots.

        Parameter
        ---------
        style: dict, optional
            dict for overwriting styles.

        Return
        ------
        fig: matplotlib.pyplot.figure
        axes: dict[matplotlib.pyplot.figure.axsubplot]
            has property "a", "b", "c", and "d".

        Example
        -------
        fig, axes = cwt.plot({"labelSize" : 20})
        axes["a"].set_ylim([-20,20])
        """

        self.style = {
            "labelSize": 16,
            "tickSize": 14,
            "titleSize": 16,
            **style
        }
        mp = MatPos()

        a = mp.add_bottom(mp, (16, 4))
        b = mp.add_bottom(a, (16, 7), offset=(0, 1), sharex=a)
        c = mp.add_right(b, (5, 7), offset=(0.5, 0), sharey=b)
        d = mp.add_bottom(b, (16, 4), offset=(0, 1), sharex=a)

        fig, axes = mp.figure_and_axes(
            subgrids) if subgrids is not None else mp.figure_and_axes([a, b, c, d])

        axes = pip(
            self.plotSignal(),
            self.plotSpector(),
            self.plotGlobalPower(),
            self.plotAverageScale(),
            self.setTextSize()
        )(axes)

        return (fig, dict(zip(["a", "b", "c", "d"], axes)))

    def setTextSize(self):
        def plot(axes):
            for ax in axes:
                ax.tick_params(axis="y", labelsize=self.style["tickSize"])
                ax.tick_params(axis="x", labelsize=self.style["tickSize"])
            return axes
        return plot

    def plotSignal(self):
        """
        Plot original signal and inversed signal.
        """
        def plot(axes):
            ax = axes[0]
            ax.plot(self.t, self.iwave, '-',
                    linewidth=2, color=[0.5, 0.5, 0.5])
            ax.plot(self.t, CWT.detrend(self.signal),
                    c='#2196f3', linestyle="-", linewidth=1.5)
            ax.set_title('a) Signal', fontsize=self.style["titleSize"])
            #ax.set_ylabel(r'{} [{}]'.format(label, units))
            ax.set_xlim([self.t.min(), self.t.max()])
            return axes
        return plot

    def plotSpector(self):
        """
        Second sub-plot, the normalized wavelet power spectrum and significance
            level contour lines and cone of influece hatched area. Note that period
            scale is logarithmic.
        """
        def plot(axes):

            bx = axes[1]
            t = self.t
            dt = self.dt
            period = self.period
            coi = self.coi

            levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
            bx.contourf(
                t, np.log2(period), np.log2(self.power), np.log2(levels),
                extend='both', cmap=plt.cm.viridis
            )
            extent = [t.min(), t.max(), 0, max(period)]

            # contour of significance
            bx.contour(
                t, np.log2(period), self.significantPower, [-99, 1],
                colors='k',
                linewidths=2,
                extent=extent
            )

            # cone of influence
            bx.fill(
                np.concatenate(
                    # [t, t[-1:] + dt, t[-1:] + dt,t[:1] - dt, t[:1] - dt]),
                    [t, t[-1:], t[-1:], t[:1], t[:1]]),
                np.concatenate(
                    [np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
                'k', alpha=0.3, hatch='x'
            )
            bx.set_ylim([np.log2(period.min()), np.log2(period.max())])
            bx.set_title('b) Wavelet Power Spectrum ',
                         fontsize=self.style["titleSize"])
            bx.set_ylabel('Period ['+self.timeUnit+']',
                          fontsize=self.style["labelSize"])

            Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                    np.ceil(np.log2(period.max())))
            bx.set_yticks(np.log2(Yticks))
            bx.set_yticklabels(Yticks)

            return axes
        return plot

    def plotGlobalPower(self):
        """
        Third sub-plot, the global wavelet and Fourier power spectra and
            theoretical noise spectra.
        Note that period scale is logarithmic.
        """
        def plot(axes):
            cx = axes[2]

            var = self.std**2
            glbl_power = self.glbl_power
            glbl_signif = self.glbl_signif
            period = self.period
            fft_theor = self.fft_theor
            fft_power = self.fft_power
            fft_freqs = self.fft_freqs

            # 95% confident spectora
            cx.plot(
                glbl_signif, np.log2(period),
                'k--'
            )

            # Red noise spectora
            cx.plot(
                var * fft_theor, np.log2(period),
                '--', color='#cccccc'
            )

            # Fourier power spectora
            cx.plot(
                var * fft_power, np.log2(1./fft_freqs),
                '-', color='#cccccc', linewidth=1.
            )

            # Global wavelet power spectora
            cx.plot(
                var * glbl_power, np.log2(period),
                'k-', linewidth=1.5
            )

            cx.set_title('c) Global Wavelet Spectrum',
                         fontsize=self.style["titleSize"])
            cx.set_xlabel(r'Power', fontsize=self.style["labelSize"])
            cx.set_xlim([0, fft_power.max() * var])
            cx.set_ylim(np.log2([period.min(), period.max()]))
            Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                    np.ceil(np.log2(period.max())))
            cx.set_yticks(np.log2(Yticks))
            cx.set_yticklabels(Yticks)
            plt.setp(cx.get_yticklabels(), visible=False)
            return axes
        return plot

    def plotAverageScale(self):
        def plot(axes):
            # Fourth sub-plot, the scale averaged wavelet spectrum.
            dx = axes[3]
            dx.axhline(
                self.scale_avg_signif,
                color='k', linestyle='--', linewidth=1.
            )
            dx.plot(
                self.t, self.scale_avg,
                'k-', linewidth=1.5
            )
            dx.set_title('d) scale-averaged power',
                         fontsize=self.style["titleSize"])
            dx.set_xlabel('Time ['+self.timeUnit+']',
                          fontsize=self.style["labelSize"])
            dx.set_ylabel(r'Average variance []',
                          fontsize=self.style["labelSize"])

            return axes
        return plot
