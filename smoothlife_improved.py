#!/usr/bin/env python3

"""
SmoothLife - Conway's Game of Life in Continuous Space

An improved implementation with modern Python practices, better performance,
and enhanced features.

Based on the work by Stephan Rafler:
https://arxiv.org/pdf/1111.1567.pdf
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import Colormap


class ThresholdType(Enum):
    """Types of threshold functions available."""
    HARD = "hard"
    LINEAR = "linear"
    LOGISTIC = "logistic"


class TimestepMode(Enum):
    """Different timestep integration modes."""
    DISCRETE = 0
    EULER_GROWTH = 1
    EULER_DECAY = 2
    MODIFIED_EULER_GROWTH = 3
    MODIFIED_EULER_DECAY = 4
    ADAMS_BASHFORTH = 5


@dataclass(frozen=True)
class RuleParameters:
    """Parameters for SmoothLife rules."""
    # Birth range
    birth_min: float = 0.278
    birth_max: float = 0.365
    
    # Survival range
    death_min: float = 0.267
    death_max: float = 0.445
    
    # Sigmoid widths
    neighbor_width: float = 0.028  # N in original
    local_width: float = 0.147     # M in original
    
    # Advanced parameters
    threshold_type: ThresholdType = ThresholdType.LOGISTIC
    mix_type: ThresholdType = ThresholdType.LOGISTIC
    timestep_mode: TimestepMode = TimestepMode.DISCRETE
    dt: float = 0.1


class ThresholdFunctions:
    """Collection of threshold and interval functions."""
    
    @staticmethod
    def logistic_threshold(x: npt.NDArray, x0: float, alpha: float) -> npt.NDArray:
        """Smooth logistic threshold function."""
        with np.errstate(over='ignore'):
            return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - x0)))
    
    @staticmethod
    def hard_threshold(x: npt.NDArray, x0: float) -> npt.NDArray:
        """Hard step threshold function."""
        return (x > x0).astype(np.float64)
    
    @staticmethod
    def linear_threshold(x: npt.NDArray, x0: float, alpha: float) -> npt.NDArray:
        """Linear threshold with transition region."""
        return np.clip((x - x0) / alpha + 0.5, 0, 1)
    
    @classmethod
    def get_threshold_func(cls, threshold_type: ThresholdType) -> Callable:
        """Get threshold function by type."""
        mapping = {
            ThresholdType.HARD: cls.hard_threshold,
            ThresholdType.LINEAR: cls.linear_threshold,
            ThresholdType.LOGISTIC: cls.logistic_threshold,
        }
        return mapping[threshold_type]
    
    @classmethod
    def interval(
        cls, 
        x: npt.NDArray, 
        a: float, 
        b: float, 
        alpha: float, 
        threshold_type: ThresholdType = ThresholdType.LOGISTIC
    ) -> npt.NDArray:
        """Interval function: returns 1 when a < x < b, 0 otherwise."""
        thresh_func = cls.get_threshold_func(threshold_type)
        if threshold_type == ThresholdType.HARD:
            return thresh_func(x, a) * (1.0 - thresh_func(x, b))
        else:
            return thresh_func(x, a, alpha) * (1.0 - thresh_func(x, b, alpha))


class ConvolutionKernels:
    """Manages convolution kernels for neighbor counting."""
    
    def __init__(
        self, 
        size: Tuple[int, int], 
        inner_radius: float = 7.0,
        outer_radius: Optional[float] = None
    ):
        self.size = size
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius or (inner_radius * 3.0)
        
        self._inner_kernel_fft: Optional[npt.NDArray] = None
        self._outer_kernel_fft: Optional[npt.NDArray] = None
        self._build_kernels()
    
    def _build_kernels(self) -> None:
        """Build and cache FFT kernels."""
        inner = self._antialiased_circle(self.inner_radius)
        outer = self._antialiased_circle(self.outer_radius)
        annulus = outer - inner
        
        # Normalize kernels
        inner = inner / np.sum(inner) if np.sum(inner) > 0 else inner
        annulus = annulus / np.sum(annulus) if np.sum(annulus) > 0 else annulus
        
        # Precompute FFTs
        self._inner_kernel_fft = np.fft.fft2(inner)
        self._outer_kernel_fft = np.fft.fft2(annulus)
    
    def _antialiased_circle(self, radius: float, roll: bool = True) -> npt.NDArray:
        """Create antialiased circle kernel."""
        h, w = self.size
        yy, xx = np.mgrid[:h, :w]
        
        # Distance from center
        distances = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        
        # Antialiasing factor
        log_res = math.log(min(h, w), 2)
        
        with np.errstate(over='ignore'):
            circle = 1 / (1 + np.exp(log_res * (distances - radius)))
        
        if roll:
            circle = np.roll(circle, h // 2, axis=0)
            circle = np.roll(circle, w // 2, axis=1)
            
        return circle
    
    def convolve(self, field: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Convolve field with inner and outer kernels."""
        field_fft = np.fft.fft2(field)
        
        inner_conv = np.real(np.fft.ifft2(field_fft * self._inner_kernel_fft))
        outer_conv = np.real(np.fft.ifft2(field_fft * self._outer_kernel_fft))
        
        return inner_conv, outer_conv


class RuleEngine(ABC):
    """Abstract base class for SmoothLife rule engines."""
    
    def __init__(self, params: RuleParameters):
        self.params = params
        self._history: list = []
    
    @abstractmethod
    def transition(
        self, 
        neighbors: npt.NDArray, 
        local: npt.NDArray, 
        field: npt.NDArray
    ) -> npt.NDArray:
        """Calculate the transition function."""
        pass
    
    def clear_history(self) -> None:
        """Clear timestep history."""
        self._history.clear()


class StandardRuleEngine(RuleEngine):
    """Standard SmoothLife rule engine with configurable parameters."""
    
    def transition(
        self, 
        neighbors: npt.NDArray, 
        local: npt.NDArray, 
        field: npt.NDArray
    ) -> npt.NDArray:
        """Apply SmoothLife transition rules."""
        p = self.params
        
        # Calculate aliveness from local cell density
        aliveness = ThresholdFunctions.get_threshold_func(p.threshold_type)(
            local, 0.5, p.local_width
        )
        
        # Interpolate birth/death thresholds based on aliveness
        threshold_min = self._lerp(p.birth_min, p.death_min, aliveness)
        threshold_max = self._lerp(p.birth_max, p.death_max, aliveness)
        
        # Apply interval function to neighbor density
        survival = ThresholdFunctions.interval(
            neighbors, threshold_min, threshold_max, 
            p.neighbor_width, p.threshold_type
        )
        
        # Apply timestep integration
        return self._integrate_timestep(survival, local, field)
    
    def _integrate_timestep(
        self, 
        survival: npt.NDArray, 
        local: npt.NDArray, 
        field: npt.NDArray
    ) -> npt.NDArray:
        """Integrate using specified timestep method."""
        mode = self.params.timestep_mode
        dt = self.params.dt
        
        if mode == TimestepMode.DISCRETE:
            return survival
        elif mode == TimestepMode.EULER_GROWTH:
            return field + dt * (2 * survival - 1)
        elif mode == TimestepMode.EULER_DECAY:
            return field + dt * (survival - field)
        elif mode == TimestepMode.MODIFIED_EULER_GROWTH:
            return local + dt * (2 * survival - 1)
        elif mode == TimestepMode.MODIFIED_EULER_DECAY:
            return local + dt * (survival - local)
        elif mode == TimestepMode.ADAMS_BASHFORTH:
            return self._adams_bashforth_step(survival, local, field)
        
        return survival
    
    def _adams_bashforth_step(
        self, 
        survival: npt.NDArray, 
        local: npt.NDArray, 
        field: npt.NDArray
    ) -> npt.NDArray:
        """Adams-Bashforth multistep integration."""
        s_current = survival - local
        self._history.insert(0, s_current)
        self._history = self._history[:4]  # Keep only last 4 steps
        
        dt = self.params.dt
        n_steps = len(self._history)
        
        if n_steps == 1:
            delta = self._history[0]
        elif n_steps == 2:
            delta = (3 * self._history[0] - self._history[1]) / 2
        elif n_steps == 3:
            delta = (23 * self._history[0] - 16 * self._history[1] + 
                    5 * self._history[2]) / 12
        else:  # n_steps >= 4
            delta = (55 * self._history[0] - 59 * self._history[1] + 
                    37 * self._history[2] - 9 * self._history[3]) / 24
        
        return field + dt * delta
    
    @staticmethod
    def _lerp(a: float, b: float, t: npt.NDArray) -> npt.NDArray:
        """Linear interpolation."""
        return (1.0 - t) * a + t * b


class SmoothLife:
    """Main SmoothLife simulation class."""
    
    def __init__(
        self, 
        width: int, 
        height: int,
        rule_params: Optional[RuleParameters] = None,
        inner_radius: float = 7.0,
        outer_radius: Optional[float] = None
    ):
        self.width = width
        self.height = height
        self.shape = (height, width)
        
        # Initialize components
        self.params = rule_params or RuleParameters()
        self.kernels = ConvolutionKernels(self.shape, inner_radius, outer_radius)
        self.rule_engine = StandardRuleEngine(self.params)
        
        # State
        self.field: npt.NDArray = np.zeros(self.shape, dtype=np.float64)
        self.generation = 0
    
    def step(self) -> npt.NDArray:
        """Advance simulation by one timestep."""
        # Convolve with kernels to get local and neighbor densities
        local_density, neighbor_density = self.kernels.convolve(self.field)
        
        # Apply transition rules
        self.field = self.rule_engine.transition(
            neighbor_density, local_density, self.field
        )
        
        # Clamp values and update generation
        self.field = np.clip(self.field, 0.0, 1.0)
        self.generation += 1
        
        return self.field.copy()
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.field.fill(0.0)
        self.generation = 0
        self.rule_engine.clear_history()
    
    def add_random_seeds(self, count: Optional[int] = None, intensity: float = 1.0) -> None:
        """Add random seed patterns to the field."""
        if count is None:
            # Estimate reasonable density based on kernel size
            area_per_seed = (self.kernels.outer_radius * 2) ** 2
            count = max(1, int(self.width * self.height / area_per_seed))
        
        radius = max(1, int(self.kernels.outer_radius))
        
        for _ in range(count):
            r = np.random.randint(0, max(1, self.height - radius))
            c = np.random.randint(0, max(1, self.width - radius))
            
            r_end = min(self.height, r + radius)
            c_end = min(self.width, c + radius)
            
            self.field[r:r_end, c:c_end] = intensity
    
    def add_pattern(self, row: int, col: int, pattern: npt.NDArray) -> None:
        """Add a specific pattern at given coordinates."""
        h, w = pattern.shape
        r_start = max(0, row)
        r_end = min(self.height, row + h)
        c_start = max(0, col)
        c_end = min(self.width, col + w)
        
        # Handle boundary conditions
        pr_start = max(0, -row)
        pr_end = pr_start + (r_end - r_start)
        pc_start = max(0, -col)
        pc_end = pc_start + (c_end - c_start)
        
        self.field[r_start:r_end, c_start:c_end] = pattern[pr_start:pr_end, pc_start:pc_end]
    
    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return {
            'generation': self.generation,
            'population': float(np.sum(self.field)),
            'alive_cells': int(np.sum(self.field > 0.5)),
            'mean_density': float(np.mean(self.field)),
            'max_density': float(np.max(self.field)),
            'min_density': float(np.min(self.field))
        }


class SmoothLifeVisualizer:
    """Handles visualization and animation of SmoothLife."""
    
    def __init__(
        self, 
        smooth_life: SmoothLife,
        colormap: Union[str, Colormap] = 'viridis',
        figsize: Tuple[float, float] = (10, 10)
    ):
        self.sim = smooth_life
        self.colormap = colormap
        self.figsize = figsize
        
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im: Optional[plt.AxesImage] = None
        self.animation: Optional[animation.FuncAnimation] = None
        
        self.paused = False
    
    def show_static(self) -> None:
        """Show current state as static image."""
        plt.figure(figsize=self.figsize)
        plt.imshow(self.sim.field, cmap=self.colormap, aspect='equal')
        plt.colorbar(label='Cell Density')
        plt.title(f'SmoothLife - Generation {self.sim.generation}')
        plt.show()
    
    def animate(
        self, 
        interval: int = 50,
        save_path: Optional[Path] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """Create and show animation."""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        self.im = self.ax.imshow(
            self.sim.field, 
            cmap=self.colormap, 
            aspect='equal',
            animated=True
        )
        
        self.ax.set_title('SmoothLife (Space: pause, R: reset, Click: add pattern)')
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        def animate_func(frame):
            if not self.paused:
                self.sim.step()
                self.im.set_array(self.sim.field)
                stats = self.sim.get_statistics()
                self.ax.set_title(
                    f"Gen: {stats['generation']}, "
                    f"Pop: {stats['population']:.1f}, "
                    f"Alive: {stats['alive_cells']}"
                )
            return [self.im]
        
        frames = None
        if duration_seconds and interval:
            frames = int(duration_seconds * 1000 / interval)
        
        self.animation = animation.FuncAnimation(
            self.fig, animate_func, frames=frames,
            interval=interval, blit=True, repeat=True
        )
        
        if save_path:
            self.animation.save(str(save_path), writer='pillow')
        else:
            plt.show()
    
    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'r':
            self.sim.reset()
            self.sim.add_random_seeds()
    
    def _on_click(self, event) -> None:
        """Handle mouse clicks."""
        if event.inaxes == self.ax and event.xdata and event.ydata:
            col = int(event.xdata)
            row = int(event.ydata)
            
            # Add a small circular pattern
            size = 10
            pattern = np.ones((size, size)) * 0.8
            self.sim.add_pattern(row - size//2, col - size//2, pattern)


# Predefined rule sets
CLASSIC_RULES = RuleParameters()

SMOOTH_RULES = RuleParameters(
    birth_min=0.254,
    birth_max=0.312,
    death_min=0.340,
    death_max=0.518,
    threshold_type=ThresholdType.LINEAR,
    timestep_mode=TimestepMode.EULER_DECAY,
    dt=0.2
)

FAST_RULES = RuleParameters(
    birth_min=0.3,
    birth_max=0.4,
    death_min=0.2,
    death_max=0.5,
    neighbor_width=0.05,
    local_width=0.1,
    dt=0.3
)


def main():
    """Example usage of the improved SmoothLife."""
    # Create simulation
    sim = SmoothLife(512, 512, rule_params=CLASSIC_RULES)
    sim.add_random_seeds(count=20)
    
    # Create visualizer and animate
    viz = SmoothLifeVisualizer(sim, colormap='plasma')
    viz.animate(interval=50)


if __name__ == "__main__":
    main()
