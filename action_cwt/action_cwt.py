import pycwt as wavelet
from pycwt.helpers import find
import numpy as np
from func_helper import pip, identity, over_args
import iter_helper as it
from structured_plot import Layout, Figure, Subplot, plot_action
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

axes_style = {
    "label": {"fontsize": 16},
    "tick": {"labelsize": 14},
    "title": {"fontsize": 16}
}

figure_padding = {
    "left": 2,
    "right": 0.5,
    "top": 0.5,
    "bottom": 1
}


def isAllTrue(arr: List[bool]) -> bool:
    return it.reducing(lambda a, b: a and b)(arr)(True)


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
        it.mapping(lambda prop: prop in obj)(props)
    )


class FourierSpectora:
    def __init__(self, fft, freqs):
        """
        parameters
        ----------
        fft: np.ndarray
        freqs: np.ndarray
        """
        self.spectora = fft
        self.freqs = freqs

    def get_specora(self):
        return self.spectora

    def get_freqs(self):
        return self.freqs

    def get_periods(self):
        return 1/self.freqs

    def get_power(self):
        return np.abs(self.spectora) ** 2


class WaveletSettings:
    def __init__(
        self,
        signal_length=0,
        dt=0,
        time_unit="",
        minimum_scale=None,
        number_of_scale=None,
        sub_octave_scale=1/2,
        mother=None,
        manual_freqs=None,
        amplitude=0
    ):
        self.signal_length = signal_length
        self.dt = dt
        self.time_unit = time_unit
        self._minimum_scale = self.dt*2 if minimum_scale is None else minimum_scale
        self.number_of_scale = number_of_scale
        self.sub_octave_scale = sub_octave_scale
        self.mother = mother
        self.manual_freqs = manual_freqs
        self.amplitude = amplitude

    def __repr__(self):
        return f"dt: {self.dt} [{self.time_unit}]\nnumber of scale: {self.J}\nminimum scale: {self.minimum_scale} [{self.time_unit}]\nsub octave scale: {self.dj}"

    @property
    def N(self):
        return self.signal_length

    @property
    def dj(self):
        return self.sub_octave_scale

    @dj.setter
    def dj(self, value):
        self.sub_octave_scale = value

    @property
    def minimum_scale(self):
        return self._minimum_scale

    @minimum_scale.setter
    def minimum_scale(self, value):
        self._minimum_scale = self.dt * 2 if value is None else value

    @property
    def J(self):
        if self.number_of_scale is None:
            return np.log2(self.signal_length*self.dt/self.minimum_scale)/self.dj
        else:
            return self.number_of_scale

    @J.setter
    def J(self, value):
        self.number_of_scale = value

    @property
    def freqs(self):
        return self.manual_freqs

    @freqs.setter
    def freqs(self, value):
        self.manual_freqs = value


class WaveletSpectora:
    def __init__(self, coeffs, scales, freqs, cone_of_influence, rectify, feature):
        self.coeffs = coeffs
        self.scales = scales
        self.freqs = freqs
        self.coi = cone_of_influence
        self.rectify = rectify
        self.feature = feature

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def get_scales(self):
        return self.scales

    def get_freqs(self):
        return self.freqs

    def get_periods(self):
        return 1/self.freqs

    def get_power(self):
        power = np.abs(self.coeffs) ** 2
        if self.rectify:
            return power / self.scales[:, None]
        else:
            return power

    def get_global_power(self):
        return self.get_power().mean(axis=1)

    def inverse(self):
        return wavelet.icwt(
            self.coeffs,
            self.scales,
            self.feature.dt,
            self.feature.dj,
            self.feature.mother
        ) * self.feature.amplitude

    def __get_significance(self, alpha, threshold, variance=1., dof=None, test_type=None):
        test_id = {
            "global": 0,
            "scale-average": 2,
        }

        def unknown_test_error(key):
            raise KeyError(
                f"Unknown test type {key}. Select from {test_id.keys()}")

        return wavelet.significance(
            variance,
            self.feature.dt,
            self.get_scales(),
            test_id.get(test_type, unknown_test_error),
            alpha,
            significance_level=threshold,
            wavelet=self.feature.mother,
            **({} if dof is None else {"dof": dof})
        )

    def get_significant_power(self, alpha, threshold):
        """
        Return
        -------
        significantPower : array like
            Significance levels as a function of scale.
        """
        significance, _ = self.__get_significance(
            alpha, threshold, test_type="global")
        mask = np.ones([1, self.feature.N]) * significance[:, None]
        return self.get_power() / mask

    def get_theoretical_fft(self, alpha, threshold):
        """
        Return
        -------
        fft_theor (array like):
            Theoretical red-noise spectrum as a function of period.

        """
        _, fft_theor = self.__get_significance(
            alpha, threshold, test_type="global")
        return fft_theor

    def get_global_significance(self, alpha, threshold):
        """
        Then, we determine significance level
        of the global wavelet spectrum.
        """
        dof = self.feature.N - self.get_scales()
        signif, _ = self.__get_significance(
            alpha,
            threshold,
            self.feature.amplitude**2,
            dof,
            test_type="global"
        )
        return signif

    def get_averaged_scale(self, period_filter):
        energy = self.feature.amplitude**2
        selector = find(period_filter(self.get_periods()))
        interval = self.feature.dj * self.feature.dt
        cdelta = self.feature.mother.cdelta

        return pip(
            lambda scales: scales.transpose(),
            lambda scales: self.get_power()/scales,
            lambda scales: scales[selector, :].sum(axis=0),
            lambda global_scale: energy * interval * global_scale / cdelta
        )(self.get_scales() * np.ones((self.feature.N, 1)))

    def get_averaged_scale_significance(self, period_filter, alpha, threshold):
        """
        Performs a scale-average test (equations 25 to 28).
        In this case dof should be set to a two element vector [s1, s2],
            which gives the scale range that were averaged together.
        If, for example, the average between scales 2 and 8 was taken, then dof=[2, 8].
        """
        selector = find(period_filter(self.get_periods()))

        scales = self.get_scales()
        dof = [scales[selector[0]], scales[selector[-1]]]
        signif, _ = self.__get_significance(
            alpha,
            threshold,
            self.feature.amplitude**2,
            dof,
            test_type="scale-average"
        )
        return signif


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

        Example
        =======
        CWT(t,x)\
            .setTimeInterval(dt)\
            .setMinimumScale(dt*2)\
            .setSubOctave(1/2)\
            .setNumberOfScale()\
            .setMotherWavelet(wavelet.Morlet(6))\
            .auto()\
            .plot()
        """

        self.t = t
        self.signal = signal

        self.dat_norm, std = pip(
            CWT.detrend if detrend else identity,
            CWT.normalized
        )(signal)
        self.period_filter = lambda period: np.array([True for v in period])
        self.verbose = verbose
        self.feature = WaveletSettings(
            signal_length=len(signal),
            amplitude=std
        )

        self.threshold = 0.95

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls)

    @staticmethod
    def setup(*arg, **kwargs):
        """
        Helper method for initiating method chain style operation.
        """
        print("CWT.setup() is deprecated. Use CWT() as constructor.")
        return CWT(*arg, **kwargs)

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
        number_of_scale: Number=None,
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

        self.feature.dt = dt
        self.feature.time_unit = unit
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
        self.feature.minimum_scale = s0
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
        self.feature.dj = 1/n
        return self

    def setNumberOfScale(self, J=None):
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

        self.feature.J = J

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
        self.feature.freqs = freqs
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
        self.feature.mother = mother
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

    def filterPeriodBy(self, pred: Callable[[VectorD], bool]=lambda period: np.array([True for v in period])):
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

        self.period_filter = pred
        return self

    def __setupCompleted(self):
        return isAllIn(self)(["dt", "dj", "s0", "mothor"])

    def cwt(self, rectify=False):
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
        wave, \
            scales, \
            freqs, \
            coi, \
            fft, \
            fft_freqs \
            = wavelet.cwt(
                self.dat_norm,
                self.feature.dt,
                self.feature.dj,
                self.feature.minimum_scale,
                self.feature.J,
                self.feature.mother,
                self.feature.freqs
            )

        wavelet_spectra = WaveletSpectora(
            wave,
            scales,
            freqs,
            coi,
            rectify,
            self.feature
        )

        fourier_spectra = FourierSpectora(
            fft,
            fft_freqs
        )

        self.wavelet_spectra = wavelet_spectra
        self.fourier_spectra = fourier_spectra

        return (
            wavelet_spectra,
            fourier_spectra
        )

    def getAlpha(self):
        """
        Lag-1 autocorrelation for red noise
        """
        self.alpha, var, mu2 = wavelet.ar1(self.signal)

        if self.verbose:
            print(
                f"Variance of signal: {self.wavelet_spectra.feature.amplitude**2}")
            print("Variance of red noise: ", var * self.feature.amplitude**2)
            print("Mean square of red noise: ",
                  mu2 * self.feature.amplitude**2)

        return self.alpha

    def auto(self, rectify=False):
        """
        Automatic operation of cwt and plotting result.
        """
        if (not self.__setupCompleted):
            raise SystemError("setup not completed.")

        self.cwt(rectify)
        self.getAlpha()

        if (self.verbose):
            print(self.feature)
            print("scale: ", self.wavelet_spectra.get_scales())
            print("period: ", self.wavelet_spectra.get_periods())

        return self

    def plot(self, layout=None, axes_style=axes_style, padding=figure_padding, **kwargs):
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

        if layout is None:
            layout = Layout()

            # Signal and inversed sugnal
            layout.add_origin("a", (10, 2))

            # Spector
            layout.add_bottom("a", "b", (10, 6), margin=1, sharex="a")

            # Global power spector
            layout.add_right("b", "c", (3, 6), margin=0.25, sharey="b")

            # Variance
            layout.add_bottom("b", "d", (10, 2), margin=1, sharex="a")

            # Mother wavelet
            layout.add_right("a", "e", (1.5, 1.5),
                             margin=0.25, offset=(0.75, 0.25))

        return Figure().add_subplot(
            self.plotSignal(Subplot(axes_style))
            + self.plotInversedSignal(Subplot(axes_style)),
            self.plotSpector(Subplot(axes_style))
            + self.plotSignificantSpector(Subplot(axes_style))
            + self.plotConeOfInfluence(Subplot(axes_style)),
            self.plotGlobalPower(
                Subplot(axes_style, ylabel={"ylabelposition": "right"})),
            self.plotAverageScale(Subplot(axes_style)),
            self.plotMotherWavelet(Subplot(axes_style))
        ).show(layout, padding=padding)

    def plotSignal(self, subplot, style={}):
        """
        Plot original signal and inversed signal.
        """

        st = {
            "color": '#2196f3',
            "linestyle": "-",
            "linewidth": 1.5,
            "alpha": 1,
            **style
        }

        return subplot.add(
            data=None,
            x=self.t,
            y=CWT.detrend(self.signal),
            plot=plot_action.line(**st),
            xlim=[self.t.min(), self.t.max()],
            title='a) Signal',
        )

    def plotInversedSignal(self, subplot, style={}):
        """
        plot inversed signal
        """

        st = {
            "color": [0.5, 0.5, 0.5],
            "linestyle": "-",
            "linewidth": 2,
            "alpha": 1,
            **style
        }

        return subplot.add(
            data=None,
            x=self.t,
            y=self.wavelet_spectra.inverse(),
            plot=plot_action.line(**st)
        )

    def plotSpector(self, subplot):
        """
        Second sub-plot, the normalized wavelet power spectrum and significance
            level contour lines and cone of influece hatched area. Note that period
            scale is logarithmic.
        """
        t = self.t
        period = self.wavelet_spectra.get_periods()
        power = self.wavelet_spectra.get_power()
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                np.ceil(np.log2(period.max())))

        return subplot.add(
            data=None,
            x=t,
            y=np.log2(period),
            z=np.log2(power),
            levels=np.log2(levels),
            cmap=plt.cm.viridis,
            plot=plot_action.contourf(extend='both'),
            ylabel=f'Period [{self.wavelet_spectra.feature.time_unit}]',
            ylim=[np.log2(period.min()), np.log2(period.max())],
            ytick={
                "locations": np.log2(Yticks),
                "labels": Yticks
            },
            title='b) Wavelet Power Spectrum '
        )

    def plotSignificantSpector(self, subplot, style={}):
        st = {
            "linewidths": 1,
            "colors": "#ffff88",
            "alpha": 1,
            **style
        }

        t = self.t
        period = self.wavelet_spectra.get_periods()
        extent = [t.min(), t.max(), 0, max(period)]

        return subplot.add(
            data=None,
            x=t,
            y=np.log2(period),
            z=self.wavelet_spectra.get_significant_power(
                self.alpha, self.threshold),
            levels=[-99, 1],
            extent=extent,
            plot=plot_action.contour(**st)
        ).add(
            data=None,
            x=t,
            y=np.log2(period),
            z=self.wavelet_spectra.get_significant_power(
                self.alpha, self.threshold),
            levels=[1, 1000],
            extent=extent,
            plot=plot_action.contourf(
                colors=["#ffffbb"], alpha=0.5,)
        )

    def plotConeOfInfluence(self, subplot, style={}):
        """
        Plot mask of cone of influence
        """
        st = {
            "color": "k",
            "alpha": 0.3,
            "hatch": "x",
            **style
        }

        t = self.t
        period = self.wavelet_spectra.get_periods()
        coi = self.wavelet_spectra.coi

        return subplot.add(
            data=None,
            x=np.concatenate([t, t[-1:], t[-1:], t[:1], t[:1]]),
            y=np.concatenate(
                [np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
            plot=plot_action.fill(**st)
        )

    def plotGlobalPower(self, subplot):
        """
        Third sub-plot, the global wavelet and Fourier power spectra and
            theoretical noise spectra.
        Note that period scale is logarithmic.
        """
        var = self.wavelet_spectra.feature.amplitude**2
        glbl_power = self.wavelet_spectra.get_global_power()
        glbl_signif = self.wavelet_spectra.get_global_significance(
            self.alpha, self.threshold)
        period = self.wavelet_spectra.get_periods()
        fft_theor = self.wavelet_spectra.get_theoretical_fft(
            self.alpha, self.threshold)
        fft_power = self.fourier_spectra.get_power()
        fft_periods = self.fourier_spectra.get_periods()
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                np.ceil(np.log2(period.max())))

        return subplot.add(
            data=None,
            x=var * fft_power,
            y=np.log2(fft_periods),
            plot=plot_action.line(
                linestyle='-', color='#cccccc', linewidth=1.),
            xlabel="Power",
            xlim=[0, fft_power.max() * var],
            ylim=np.log2([period.min(), period.max()]),
            tick=dict(labelleft=False, left=False,
                      labelright=True, right=True),
            ytick={
                "locations": np.log2(Yticks),
                "labels": Yticks
            },
            title='c) Global Wavelet Spectrum',
        ).add(
            data=None,
            x=var * glbl_power,
            y=np.log2(period),
            plot=plot_action.line(
                linestyle='-', color="#1146b3", linewidth=1.5)
        ).add(
            data=None,
            x=glbl_signif,
            y=np.log2(period),
            plot=plot_action.line(linestyle='--', color="#2196f3")
        ).add(
            data=None,
            x=var * fft_theor,
            y=np.log2(period),
            plot=plot_action.line(linestyle='--', color='#cccccc'),
            xscale="log",
        )

    def plotAverageScale(self, subplot):
        return subplot.add(
            data=None,
            ypos=self.wavelet_spectra.get_averaged_scale_significance(
                self.period_filter, self.alpha, self.threshold),
            plot=plot_action.xband(color='k', linestyle='--', linewidth=1.),
            xlabel='Time ['+self.wavelet_spectra.feature.time_unit+']',
            ylabel='Average variance []',
            title='d) scale-averaged power',
        ).add(
            data=None,
            x=self.t,
            y=self.wavelet_spectra.get_averaged_scale(self.period_filter),
            plot=plot_action.line(color="k", linewidth=1.5)
        )

    def plotMotherWavelet(self, subplot):
        t = np.arange(-5, 5, 0.01)
        return subplot.add(
            data=None,
            x=t,
            y=[self.wavelet_spectra.feature.mother.psi(x)for x in t],
            plot=plot_action.line(color="k"),
            tick=dict(bottom=False, left=False,
                      labelbottom=False, labelleft=False),
            title="Mother wavelet"
        )
