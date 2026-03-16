try:
    from .barnes_hut_cython import BarnesHut, Particle, Rectangle, integrate_step_1, integrate_step_2
except ImportError:
    raise ImportError("Cython module not found. Please compile barnes_hut_cython.pyx before running the simulation.")
import numpy as np
import os
import time

try:
    from progress.bar import Bar
except ImportError:
    class Bar:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def next(self):
            pass


class NBodySimulation:
    def __init__(
        self,
        particles: list[Particle],
        boundary: Rectangle,
        theta: float = 0.8,
        softening: float = 0.1,
        max_capacity: int = 8,
        tree_rebuild_interval: int = 1,
    ) -> None:
        self.particles = particles
        self.boundary = boundary
        self.theta = theta
        self.softening = softening
        self.max_capacity = max_capacity
        self.tree_rebuild_interval = tree_rebuild_interval
        self.current_step = 0
        self.tree: BarnesHut | None = None
        
    def build_tree(self) -> None:
        self.tree = BarnesHut(self.boundary, max_capacity=self.max_capacity)
        for particle in self.particles:
            self.tree.insert(particle)
            
    def _compute_forces(self) -> None:
        # Only rebuild tree at specified intervals
        if self.tree is None or self.current_step % self.tree_rebuild_interval == 0:
            self.build_tree()
            
        if self.tree is None:
            for p in self.particles:
                p.fx = 0.0
                p.fy = 0.0
            return
        
        tree = self.tree
        for p in self.particles:
            tree.compute_force(p, self.theta, self.softening)
            
    def step(self, dt: float) -> None:
        if not self.particles:
            return
        
        if self.current_step == 0:
            self._compute_forces()            
        
        # First half of the leapfrog integration
        integrate_step_1(self.particles, dt)
        # Compute new forces after the first half step
        new_forces = self._compute_forces()
        # Second half of the leapfrog integration
        integrate_step_2(self.particles, dt)
    
        self.current_step += 1
                
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
        if save_interval > 0 and save_path == '':
            save_path = f'saves_{int(time.time())}/'
            
        if save_interval > 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Export the initial state before starting the simulation and settings
        if save_interval > 0:
            self.save_state(0, save_path)
            t_final = steps * dt
            self.export_settings(save_path, dt, t_final)
        
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
        np.savez(f"{save_path}state_{step:06d}.npz", 
                 x=[p.x for p in self.particles], 
                 y=[p.y for p in self.particles], 
                 vx=[p.vx for p in self.particles], 
                 vy=[p.vy for p in self.particles], 
                 mass=[p.mass for p in self.particles])
        
    def export_settings(self, save_path: str, dt: float, t_final: float) -> None:
        """
        Exports the simulation settings to a file for later reference.
        
        Parameters:
        - save_path (str): The directory path where the settings will be saved.
        - dt (float): The time interval for each step.
        - t_final (float): The final time of the simulation.
        """
        settings = {
            'theta': self.theta,
            'softening': self.softening,
            'max_capacity': self.max_capacity,
            'tree_rebuild_interval': self.tree_rebuild_interval,
            'boundary': (self.boundary.x_min, self.boundary.x_max, self.boundary.y_max, self.boundary.y_min),
            'dt': dt,
            't_final': t_final,
        }
        np.savez(f"{save_path}settings.npz", **settings)