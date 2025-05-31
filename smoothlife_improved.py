#!/usr/bin/env python3
"""
SmoothLife (PyFFTW Edition)

Uses pyFFTW for accelerated FFTs, with float32 arrays and real-to-complex transforms.
Comments focus on explaining non-obvious parts of the implementation.
"""

import argparse
import math
import time

import numpy as np
import pyfftw

# Enable pyFFTW's internal wisdom/cache to reuse plans and improve performance on repeated calls.
pyfftw.interfaces.cache.enable()

# Import rfft2 and irfft2 from pyFFTW's NumPy FFT interface.
from pyfftw.interfaces.numpy_fft import rfft2, irfft2

from matplotlib import pyplot as plt
from matplotlib import animation


def logistic_threshold(x: np.ndarray, x0: float, alpha: float) -> np.ndarray:
    """
    Smooth approximation to a step function centered at x0 with width determined by alpha.
    The factor -4/alpha ensures the transition region spans roughly alpha in width.

    Returns values in (0,1), approaching 0 for x << x0 and 1 for x >> x0.
    """
    return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - x0)))


def hard_threshold(x: np.ndarray, x0: float) -> np.ndarray:
    """
    Sharp cutoff: returns 1.0 where x > x0, else 0.0.
    Used when an exact, non-smooth boundary is desired.
    """
    return np.greater(x, x0).astype(np.float32)


def linearized_threshold(x: np.ndarray, x0: float, alpha: float) -> np.ndarray:
    """
    Linear ramp: transitions linearly from 0 to 1 over a window of width alpha centered at x0.
    Outside that window, it clamps to 0 or 1.
    """
    return np.clip((x - x0) / alpha + 0.5, 0.0, 1.0)


def logistic_interval(x: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    """
    Smooth indicator for values between a and b.
    Combines two logistic thresholds to produce something near 1 when a < x < b,
    and near 0 outside, with smooth edges of width ~alpha.
    """
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha))


def linearized_interval(x: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    """
    Smooth indicator using linear ramps: rises from 0 to 1 around a, stays near 1,
    then falls from 1 to 0 around b. The total "transition" width around each boundary is alpha.
    """
    return linearized_threshold(x, a, alpha) * (1.0 - linearized_threshold(x, b, alpha))


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Linear interpolation between arrays a and b, weighted by t.
    All arrays should be float32; this computes (1 - t)*a + t*b elementwise.
    """
    return (1.0 - t) * a + t * b


class BasicRules:
    """
    Defines the core birth (B1,B2) and survival (D1,D2) thresholds, plus sigmoid widths N and M.
    Computes the next "aliveness" value as a float between 0 and 1.
    """

    B1 = 0.278
    B2 = 0.365
    D1 = 0.267
    D2 = 0.445

    # Widths controlling how gradual the logistic transitions are
    N = 0.028
    M = 0.147

    def __init__(self, **kwargs):
        # Allow overriding any of the parameters via keyword arguments
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))
            else:
                raise ValueError(f"Unexpected BasicRules parameter: '{k}'")

    def clear(self):
        # No internal state to reset for BasicRules
        pass

    def s(self, n: np.ndarray, m: np.ndarray, field: np.ndarray) -> np.ndarray:
        """
        Takes:
          - n: neighbor integral over annulus (float32 array)
          - m: neighbor integral over inner circle (float32 array)
          - field: current field (float32 array of values in [0,1])

        Returns the next field, also values in [0,1].
        """
        # Compute a smooth "aliveness" metric from m. When m ≈ 0.5, logistic_threshold(m,0.5,M) ≈ 0.5.
        aliveness = logistic_threshold(m, 0.5, self.M).astype(np.float32)

        # Interpolate between birth and death thresholds based on how "alive" the cell already is.
        threshold1 = lerp(self.B1, self.D1, aliveness)  # lower bound
        threshold2 = lerp(self.B2, self.D2, aliveness)  # upper bound

        # Now compute a smooth indicator of whether n is between [threshold1, threshold2].
        new_aliveness = logistic_interval(n, threshold1, threshold2, self.N)

        # Clamp to [0,1] as a safety check
        return np.clip(new_aliveness, 0.0, 1.0).astype(np.float32)


class ExtensiveRules(BasicRules):
    """
    Extends BasicRules with additional modes for sigmode (choosing how thresholds are applied),
    sigtype (how to compute intervals), mixtype (how to mix birth/death results), and multiple
    timestep_mode options including Adams-Bashforth integration.
    """

    sigmode = 0
    sigtype = 0
    mixtype = 0
    timestep_mode = 0
    dt = 0.1

    def __init__(self, **kwargs):
        # Allow overriding any class-level attribute via kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v) if isinstance(v, (int, float)) else v)
            else:
                raise ValueError(f"Unexpected ExtensiveRules parameter: '{k}'")

        # ess array holds up to 3 previous "s" values for Adams-Bashforth mode
        self.esses = [None] * 3
        self.esses_count = 0

    def clear(self):
        """
        Reset any stored timestepping history used for Adams-Bashforth integration
        (timestep_mode == 5).
        """
        self.esses = [None] * 3
        self.esses_count = 0

    def sigmoid_ab(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Compute an indicator of x being between a and b, depending on sigtype:

        - sigtype == 0: hard threshold (binary).
        - sigtype == 1: linearized interval.
        - sigtype == 4: logistic interval.

        Raises if sigtype is not recognized.
        """
        if self.sigtype == 0:
            return hard_threshold(x, a) * (1 - hard_threshold(x, b))
        elif self.sigtype == 1:
            return linearized_interval(x, a, b, self.N)
        elif self.sigtype == 4:
            return logistic_interval(x, a, b, self.N)
        else:
            raise NotImplementedError(f"sigtype {self.sigtype} not implemented")

    def sigmoid_mix(self, x: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Mix two arrays x and y based on the "aliveness" metric m, using mixtype:

        - mixtype == 0: binary threshold on m.
        - mixtype == 1: linearized threshold on m.
        - mixtype == 4: logistic threshold on m.

        Returns a linear interpolation (lerp) between x and y weighted by the chosen mix.
        """
        if self.mixtype == 0:
            intermediate = hard_threshold(m, 0.5)
        elif self.mixtype == 1:
            intermediate = linearized_threshold(m, 0.5, self.M)
        elif self.mixtype == 4:
            intermediate = logistic_threshold(m, 0.5, self.M)
        else:
            raise NotImplementedError(f"mixtype {self.mixtype} not implemented")

        return lerp(x, y, intermediate)

    def s(self, n: np.ndarray, m: np.ndarray, field: np.ndarray) -> np.ndarray:
        """
        Compute the next field using more elaborate rules:

        1. Depending on sigmode, compute a "transition" array using different combinations
           of birth/death thresholds, mixing, and logistics.
        2. Depending on timestep_mode, integrate either discretely, via Euler, via ODE on m,
           or via Adams-Bashforth 4th order (timestep_mode == 5).

        Finally, clamp output to [0,1].
        """
        # STEP 1: Compute "transition" based on sigmode
        if self.sigmode == 1:
            b_thresh = self.sigmoid_ab(n, self.B1, self.B2)
            d_thresh = self.sigmoid_ab(n, self.D1, self.D2)
            transition = lerp(b_thresh, d_thresh, m)
        elif self.sigmode == 2:
            b_thresh = self.sigmoid_ab(n, self.B1, self.B2)
            d_thresh = self.sigmoid_ab(n, self.D1, self.D2)
            transition = self.sigmoid_mix(b_thresh, d_thresh, m)
        elif self.sigmode == 3:
            threshold1 = lerp(self.B1, self.D1, m)
            threshold2 = lerp(self.B2, self.D2, m)
            transition = self.sigmoid_ab(n, threshold1, threshold2)
        elif self.sigmode == 4:
            threshold1 = self.sigmoid_mix(self.B1, self.D1, m)
            threshold2 = self.sigmoid_mix(self.B2, self.D2, m)
            transition = self.sigmoid_ab(n, threshold1, threshold2)
        else:
            raise NotImplementedError(f"sigmode {self.sigmode} not implemented")

        # STEP 2: Integrate based on timestep_mode
        if self.timestep_mode == 0:
            # Discrete update: simply set next_field = transition
            next_field = transition

        elif self.timestep_mode == 1:
            # Euler-like: field ← field + dt * (2*transition - 1)
            next_field = field + self.dt * (2 * transition - 1)

        elif self.timestep_mode == 2:
            # Exponential approach: field ← field + dt * (transition - field)
            next_field = field + self.dt * (transition - field)

        elif self.timestep_mode == 3:
            # ODE on 'm' instead of 'field': m_new = m + dt*(2*transition - 1)
            next_field = m + self.dt * (2 * transition - 1)

        elif self.timestep_mode == 4:
            # ODE on 'm': m_new = m + dt*(transition - m)
            next_field = m + self.dt * (transition - m)

        elif self.timestep_mode == 5:
            # Adams-Bashforth 4-step: uses s = transition - m to estimate next field
            s0 = transition - m
            s1, s2, s3 = self.esses

            if self.esses_count == 0:
                delta = s0
            elif self.esses_count == 1:
                delta = (3 * s0 - s1) / 2
            elif self.esses_count == 2:
                delta = (23 * s0 - 16 * s1 + 5 * s2) / 12
            else:
                # When esse_count >= 3, use the 4-step Adams-Bashforth formula
                delta = (55 * s0 - 59 * s1 + 37 * s2 - 9 * s3) / 24

            # Rotate the history buffer: newest s0 enters at index 0, oldest drops off
            self.esses = [s0, s1, s2]
            if self.esses_count < 3:
                self.esses_count += 1

            next_field = field + self.dt * delta

        else:
            raise NotImplementedError(
                f"timestep_mode {self.timestep_mode} not implemented"
            )

        # Ensure the field stays within [0,1]
        return np.clip(next_field, 0.0, 1.0).astype(np.float32)


class SmoothTimestepRules(ExtensiveRules):
    """
    A convenience preset: sets parameters to specific values corresponding to one of the
    'smooth' rule sets commonly used in SmoothLife experiments.
    """

    B1 = 0.254
    B2 = 0.312
    D1 = 0.340
    D2 = 0.518

    sigmode = 2
    sigtype = 1
    mixtype = 0
    timestep_mode = 2
    dt = 0.2


def antialiased_circle(size: tuple[int, int], radius: float) -> np.ndarray:
    """
    Generate a smooth circular mask of the given radius on a grid of shape size=(h,w).

    We compute the distance from each pixel to the circle center, then apply a logistic
    function so that pixels right at the boundary (distance ≈ radius) are blurred.

    The 'roll' step recenters the circle so that its peak is at (0,0) in the frequency domain,
    making kernels symmetric for convolution via FFT.

    Returns a float32 array of shape (h,w) with values in [0,1].
    """
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    # Distance from the center of the grid
    radii = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)

    # Use a steepness roughly log2(min(h,w)) so that edge thickness scales with grid size
    logres = math.log(min(h, w), 2.0)
    # logistic here smooths the transition over a narrow band around radius
    with np.errstate(over="ignore"):
        smooth_circle = 1.0 / (1.0 + np.exp(logres * (radii - radius)))

    smooth_circle = smooth_circle.astype(np.float32)
    # Roll shifts the peak to (0,0) so that the convolution is centered when we do FFT
    smooth_circle = np.roll(smooth_circle, h // 2, axis=0)
    smooth_circle = np.roll(smooth_circle, w // 2, axis=1)
    return smooth_circle


class Multipliers:
    """
    Precompute frequency-domain kernels using real-to-complex FFT (rfft2). We maintain:
      - _M_freq: FFT of the inner radius kernel (complex64 array of shape (h, w//2+1))
      - _N_freq: FFT of the annulus kernel (outer minus inner, normalized)

    These frequency-domain arrays are used for rapid convolution: multiplying
    the FFT of the current field by these kernels and then performing inverse FFT
    yields neighborhood integrals needed by the rules.
    """

    INNER_RADIUS = 7.0
    OUTER_RADIUS = INNER_RADIUS * 3.0

    def __init__(
        self,
        size: tuple[int, int],
        inner_radius: float = None,
        outer_radius: float = None,
    ):
        h, w = size
        if inner_radius is None:
            inner_radius = Multipliers.INNER_RADIUS
        if outer_radius is None:
            outer_radius = Multipliers.OUTER_RADIUS

        # Build float32 spatial kernels
        inner = antialiased_circle((h, w), inner_radius)
        outer = antialiased_circle((h, w), outer_radius)
        annulus = outer - inner

        # Normalize each kernel so their sum is 1. This makes them proper averaging filters.
        inner /= np.sum(inner, dtype=np.float32)
        annulus /= np.sum(annulus, dtype=np.float32)

        # Compute real-to-complex FFTs. Inputs are float32; outputs are complex64.
        self._M_freq = rfft2(inner)  # shape = (h, w//2 + 1)
        self._N_freq = rfft2(annulus)  # shape = (h, w//2 + 1)

        # Record shapes for later buffer allocations
        self.freq_shape = self._M_freq.shape  # (h, w//2+1)
        self.real_shape = (h, w)

    @property
    def M(self) -> np.ndarray:
        return self._M_freq

    @property
    def N(self) -> np.ndarray:
        return self._N_freq


class SmoothLife:
    """
    Main simulation class. Holds:
      - width, height
      - float32 field array of shape (h,w)
      - Multipliers instance with precomputed FFT kernels (_M_freq, _N_freq)
      - rule instance (BasicRules, ExtensiveRules, or SmoothTimestepRules)
      - pre-allocated buffers for FFT and inverse FFT to avoid reallocations each frame
    """

    def __init__(
        self, height: int, width: int, rule_name: str = "basic", **rule_kwargs
    ):
        self.height = height
        self.width = width

        # Set up frequency-domain convolution kernels
        self.multipliers = Multipliers((height, width))

        # Choose rule set
        rule_name = rule_name.lower()
        if rule_name == "basic":
            self.rules = BasicRules(**rule_kwargs)
        elif rule_name == "extensive":
            self.rules = ExtensiveRules(**rule_kwargs)
        elif rule_name == "smooth":
            self.rules = SmoothTimestepRules(**rule_kwargs)
        else:
            raise ValueError(
                f"Unknown rule '{rule_name}'. Choose from basic, extensive, smooth."
            )

        # Zero-initialize the field array as float32
        self.field = np.zeros((height, width), dtype=np.float32)
        self.rules.clear()

        # Pre-allocate buffers:
        # _freq_buffer holds the half-complex spectrum (complex64) of shape (h, w//2 + 1).
        # _real_buffer holds an (h, w) float32 array to store inverse FFTs for the inner kernel.
        freq_h, freq_w = self.multipliers.freq_shape
        self._freq_buffer = pyfftw.empty_aligned((freq_h, freq_w), dtype="complex64")
        self._real_buffer = pyfftw.empty_aligned((height, width), dtype="float32")

        # We keep a second float32 buffer for the annulus inverse FFT
        self._real_buffer2 = pyfftw.empty_aligned((height, width), dtype="float32")

    def clear(self):
        """
        Zero out the field and reset any internal rule state.
        """
        self.field.fill(0.0)
        self.rules.clear()

    def step(self) -> np.ndarray:
        """
        Perform one simulation timestep:

          1. Compute real-to-complex FFT of self.field into _freq_buffer.
          2. Multiply that spectrum by the inner-kernel spectrum (self.multipliers.M) to get M_freq_mul.
          3. Multiply by the annulus-kernel spectrum (self.multipliers.N) to get N_freq_mul.
          4. Inverse FFT M_freq_mul into _real_buffer to obtain the inner-circle integral M_buffer.
          5. Inverse FFT N_freq_mul into _real_buffer2 to obtain the annulus integral N_buffer.
          6. Call self.rules.s(N_buffer, M_buffer, self.field) to get the next field.
          7. Return the updated field.

        Uses out= arguments to avoid allocating new arrays each call.
        """
        # Forward real-to-complex FFT: takes float32 input of shape (h,w),
        # produces complex64 output in _freq_buffer of shape (h, w//2+1).
        self._freq_buffer = rfft2(self.field, overwrite_input=False)

        # Multiply in frequency domain to get the inner-circle convolution
        M_freq_mul = self._freq_buffer * self.multipliers.M
        # Multiply in frequency domain to get the annulus convolution
        N_freq_mul = self._freq_buffer * self.multipliers.N

        # Inverse FFT for inner circle: goes from complex64 (h, w//2+1) → float32 (h, w)
        try:
            self._real_buffer = irfft2(
                M_freq_mul,
                s=self.multipliers.real_shape,
            )
            M_buffer = self._real_buffer
        except TypeError:
            M_buffer = irfft2(M_freq_mul, s=self.multipliers.real_shape).astype(
                np.float32
            )

        # Inverse FFT for annulus: store in second real buffer
        try:
            self._real_buffer2 = irfft2(
                N_freq_mul,
                s=self.multipliers.real_shape,
            )
            N_buffer = self._real_buffer2
        except TypeError:
            N_buffer = irfft2(N_freq_mul, s=self.multipliers.real_shape).astype(
                np.float32
            )

        # Compute next field values using the selected rules
        self.field = self.rules.s(N_buffer, M_buffer, self.field)
        return self.field

    def add_speckles(self, count: int = None, intensity: float = 1.0):
        """
        Scatter random square 'speckles' of alive cells across the field.
        If count is None, choose based on the kernel's outer radius so that
        speckles occur roughly once per (2*outer_radius)^2 area.
        """
        if count is None:
            default_area = (self.multipliers.OUTER_RADIUS * 2) ** 2
            count = int(self.width * self.height / default_area)

        for _ in range(count):
            # Pick a random top-left corner for a patch of size ~outer_radius
            patch_size = int(self.multipliers.OUTER_RADIUS)
            r = np.random.randint(0, self.height - patch_size)
            c = np.random.randint(0, self.width - patch_size)
            self.field[r : r + patch_size, c : c + patch_size] = intensity

    def snapshot(self, filename: str):
        """
        Save the current field as a grayscale PNG image (values in [0,1]).
        """
        plt.imsave(filename, self.field, cmap="viridis", vmin=0, vmax=1)

    def show_interactive(self):
        """
        Launch a Matplotlib window for interactive simulation:

          - Left-click: add a small cross-shaped cluster of "alive" cells at click.
          - Spacebar: pause/unpause the simulation.
          - 'R' key: reset/clear the field.
          - 'Q' key: quit the window.

        The animation updates roughly every 60 ms (~16 FPS).
        """
        sl = self
        sl.add_speckles()  # start with some initial noise

        fig, ax = plt.subplots()
        im = ax.imshow(sl.field, animated=True, cmap="viridis", aspect="equal")

        def animate(_):
            # If not paused, advance one timestep and update the image
            if not getattr(sl, "paused", False):
                arr = sl.step()
                im.set_array(arr)
            return (im,)

        def on_click(event):
            # Only add cells if the click is inside the axes
            if event.inaxes == ax:
                col = int(event.xdata)
                row = int(event.ydata)
                sl._add_cells_at(row, col)

        def on_key(event):
            key = event.key.lower()
            if key == "r":
                sl.clear()
            elif key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        ani = animation.FuncAnimation(fig, animate, interval=60, blit=True)
        plt.title("Click to add cells. Space to pause. 'R' to reset. 'Q' to quit.")
        plt.show()

    def _add_cells_at(self, row: int, col: int):
        """
        Create a plus-shaped cluster around (row, col). Each arm is cell_size pixels long.
        Ensures cells stay within the grid boundaries by clamping indices.
        """
        cell_size = getattr(self, "cell_size", 10)
        half = cell_size // 2
        positions = [
            (row, col),
            (row - cell_size, col),
            (row + cell_size, col),
            (row, col - cell_size),
            (row, col + cell_size),
        ]
        for r, c in positions:
            r0 = max(0, r - half)
            r1 = min(self.height, r + half)
            c0 = max(0, c - half)
            c1 = min(self.width, c + half)
            if 0 <= r0 < r1 <= self.height and 0 <= c0 < c1 <= self.width:
                self.field[r0:r1, c0:c1] = 1.0


def save_animation_to_file(
    height: int, width: int, rule_name: str, steps: int, output_filename: str
):
    """
    Run the simulation for a fixed number of frames, capturing each frame to an MP4 file.

    We use Matplotlib's FFMpegWriter. The figure is created off-screen (no interactive display).
    """
    sl = SmoothLife(height, width, rule_name=rule_name)
    sl.add_speckles()

    fig = plt.figure(figsize=(5, 5), dpi=100)
    im = plt.imshow(sl.field, animated=True, cmap="viridis", vmin=0, vmax=1)
    plt.axis("off")  # hide axis for a cleaner video

    from matplotlib.animation import FFMpegWriter

    # fps controls how many frames per second in the output video.
    fps = 10
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="SmoothLife"), bitrate=1800)
    with writer.saving(fig, output_filename, dpi=100):
        for _ in range(steps):
            arr = sl.step()
            im.set_array(arr)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved animation to '{output_filename}' ({steps} frames).")


def benchmark(height: int, width: int, rule_name: str, steps: int):
    """
    Run 'steps' calls to sl.step() in a tight loop (no rendering) and measure elapsed time.

    Reports total seconds and steps per second. Useful for comparing different FFT backends.
    """
    print(f"Benchmarking: {height}×{width}, rule='{rule_name}', steps={steps}")
    sl = SmoothLife(height, width, rule_name=rule_name)
    sl.add_speckles()

    start = time.time()
    for _ in range(steps):
        sl.step()
    end = time.time()

    total = end - start
    print(f"Completed {steps} steps in {total:.3f} s → {steps / total:.2f} steps/sec")


def parse_args():
    parser = argparse.ArgumentParser(description="SmoothLife with pyFFTW acceleration.")
    parser.add_argument(
        "--height", type=int, default=512, help="Grid height in pixels."
    )
    parser.add_argument("--width", type=int, default=512, help="Grid width in pixels.")
    parser.add_argument(
        "--rule",
        type=str,
        default="basic",
        choices=["basic", "extensive", "smooth"],
        help="Rule set to use: basic, extensive, or smooth.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of timesteps (used by --save-mp4 or --benchmark).",
    )
    parser.add_argument(
        "--save-mp4",
        type=str,
        default=None,
        help="If provided, save an MP4 animation to this filename.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (no visualization), performing --steps iterations.",
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Matplotlib colormap for display."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible speckles."
    )
    parser.add_argument(
        "--B1", type=float, default=None, help="Override birth lower bound B1."
    )
    parser.add_argument(
        "--B2", type=float, default=None, help="Override birth upper bound B2."
    )
    parser.add_argument(
        "--D1", type=float, default=None, help="Override death lower bound D1."
    )
    parser.add_argument(
        "--D2", type=float, default=None, help="Override death upper bound D2."
    )
    parser.add_argument(
        "--N", type=float, default=None, help="Override sigmoid width N."
    )
    parser.add_argument(
        "--M", type=float, default=None, help="Override sigmoid width M."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Collect any rule overrides into a dictionary
    rule_kwargs = {}
    for param in ["B1", "B2", "D1", "D2", "N", "M"]:
        val = getattr(args, param)
        if val is not None:
            rule_kwargs[param] = val

    if args.benchmark:
        benchmark(args.height, args.width, args.rule, args.steps)
        return

    if args.save_mp4:
        save_animation_to_file(
            args.height, args.width, args.rule, args.steps, args.save_mp4
        )
        return

    # Otherwise, launch the interactive window
    print(
        "Interactive SmoothLife: Click to add cells, space to pause, 'R' to reset, 'Q' to quit."
    )
    SL = SmoothLife(args.height, args.width, rule_name=args.rule, **rule_kwargs)
    SL.cell_size = 10
    SL.cmap = args.cmap
    SL.show_interactive()


if __name__ == "__main__":
    main()
