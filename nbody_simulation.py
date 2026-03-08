from barnes_hut import BarnesHut, Particle, Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from progress.bar import Bar

class NBodySimulation:
    def __init__(self, particles: list[Particle], boundary: Rectangle, theta: float = 0.5, softening: float = 0.1) -> None:
        self.particles = particles
        self.boundary = boundary
        self.theta = theta
        self.softening = softening
        self.tree: BarnesHut | None = None
        
    def build_tree(self) -> None:
        self.tree = BarnesHut(self.boundary)
        for particle in self.particles:
            self.tree.insert(particle)
            
    def step(self, dt: float) -> None:
        self.build_tree()
        if self.tree is None:
            return

        tree = self.tree
        # Compute forces and update velocities using explicit Euler integration
        for particle in self.particles:
            force_x, force_y = tree.compute_force(particle, self.theta, self.softening)
            particle.vx += (force_x / particle.mass) * dt
            particle.vy += (force_y / particle.mass) * dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
                
    def run(self, steps: int, dt: float, save_interval: int = -1, save_path: str = 'saves/') -> None:
        """
        Runs the N-body simulation for a given number of steps and time interval.
        
        Parameters:
        - steps (int): The number of steps to run the simulation.
        - dt (float): The time interval for each step.
        - save_interval (int): The interval at which to save the state of the simulation. If -1, no saving is done.
        - save_path (str): The directory path where the simulation states will be saved if save
        """
        if save_interval > 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        with Bar('Running Simulation', max=steps) as bar:
            for i in range(steps):
                self.step(dt)
                if save_interval > 0 and i % save_interval == 0:
                    self.save_state(i, save_path)
                bar.next()
                
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