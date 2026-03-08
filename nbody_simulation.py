from barnes_hut import BarnesHut, Particle, Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from progress.bar import Bar
import time

class NBodySimulation:
    def __init__(
        self,
        particles: list[Particle],
        boundary: Rectangle,
        theta: float = 0.8,
        softening: float = 0.1,
        max_capacity: int = 8,
    ) -> None:
        self.particles = particles
        self.boundary = boundary
        self.theta = theta
        self.softening = softening
        self.max_capacity = max_capacity
        self.tree: BarnesHut | None = None
        self._cached_forces: list[tuple[float, float]] | None = None
        
    def build_tree(self) -> None:
        self.tree = BarnesHut(self.boundary, max_capacity=self.max_capacity)
        for particle in self.particles:
            self.tree.insert(particle)
            
    def _compute_forces(self) -> list[tuple[float, float]]:
        self.build_tree()
        if self.tree is None:
            return [(0.0, 0.0) for _ in self.particles]
        tree = self.tree
        return [tree.compute_force(p, self.theta, self.softening) for p in self.particles]
            
    def step(self, dt: float) -> None:
        if not self.particles:
            return

        if self._cached_forces is None:
            self._cached_forces = self._compute_forces()

        for particle, (fx, fy) in zip(self.particles, self._cached_forces):
            vx_half = particle.vx + 0.5 * fx / particle.mass * dt
            vy_half = particle.vy + 0.5 * fy / particle.mass * dt
            particle.x += vx_half * dt
            particle.y += vy_half * dt

            particle.vx = vx_half
            particle.vy = vy_half

        new_forces = self._compute_forces()

        for particle, (fx, fy) in zip(self.particles, new_forces):
            particle.vx += 0.5 * fx / particle.mass * dt
            particle.vy += 0.5 * fy / particle.mass * dt

        self._cached_forces = new_forces
                
    def run(
        self,
        steps: int,
        dt: float,
        save_interval: int = -1,
        save_path: str = '',
        show_progress: bool = True,
    ) -> None:
        """
        Runs the N-body simulation for a given number of steps and time interval.
        
        Parameters:
        - steps (int): The number of steps to run the simulation.
        - dt (float): The time interval for each step.
        - save_interval (int): The interval at which to save the state of the simulation. If -1, no saving is done.
        - save_path (str): The directory path where the simulation states will be saved if save_interval is specified.
        - show_progress (bool): Whether to display a progress bar during the simulation.
        """
        if save_interval > 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if save_interval > 0 and save_path == '':
            save_path = f'saves_{int(time.time())}/'
        
        if show_progress:
            with Bar('Running Simulation', max=steps) as bar:
                for i in range(steps):
                    self.step(dt)
                    if save_interval > 0 and i % save_interval == 0:
                        self.save_state(i, save_path)
                    bar.next()
            return

        for i in range(steps):
            self.step(dt)
            if save_interval > 0 and i % save_interval == 0:
                self.save_state(i, save_path)
                
    def save_state(self, step: int, save_path: str) -> None:
        """
        Saves the current state of the simulation to a file.
        
        Parameters:
        - step (int): The current step number, used for naming the save file.
        - save_path (str): The directory path where the simulation state will be saved.
        """
        np.savez(f"{save_path}state_{step}.npz", 
                 x=[p.x for p in self.particles], 
                 y=[p.y for p in self.particles], 
                 vx=[p.vx for p in self.particles], 
                 vy=[p.vy for p in self.particles], 
                 mass=[p.mass for p in self.particles])