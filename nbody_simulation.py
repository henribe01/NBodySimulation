from barnes_hut import BarnesHut, Particle, Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
                
    def run(self, steps: int, dt: float) -> None:
        """
        Runs the N-body simulation for a given number of steps and time interval.
        
        Parameters:
        - steps (int): The number of steps to run the simulation.
        - dt (float): The time interval for each step.
        """
        for i in range(steps):
            self.step(dt)