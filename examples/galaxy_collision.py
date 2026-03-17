from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from nbody.nbody_simulation import NBodySimulation, Particle, Rectangle
import matplotlib.pyplot as plt
from galaxy import generate_particles

# Galaxy 1
galaxy1_particles = generate_particles(
    num_stars=8000,
    center=(-3.0, 0.5),
    initial_velocity=(0.4, 0.0),
    min_radius=0.2,
    max_radius=2.0,
    mass_bulge=1.0,
    mass_disk=0.3
)
# Galaxy 2
galaxy2_particles = generate_particles(
    num_stars=4000,
    center=(3.0, -0.5),
    initial_velocity=(-0.4, 0.0),
    min_radius=0.1,
    max_radius=1.0,
    mass_bulge=0.5,
    mass_disk=0.15
)
particles = galaxy1_particles + galaxy2_particles

# Simulation settings
softening = 0.05
dt = 0.01
boundary = Rectangle(-20, 20, 20, -20)
max_capacity = 4
theta = 0.8
rebuild_interval = 1
t_final = 60.0
print(f"Running simulation for {t_final:.2f} time units with dt={dt:.2f} time units.")

# Save interval for output
fps = 30  # Desired frames per second for output
animation_time = 30  # Total time for the animation in seconds
save_interval = max(1, int((t_final / (fps * animation_time)) / dt))  # Save every few steps to achieve desired fps and animation length
print(f"Saving output every {save_interval} steps (every {save_interval * dt:.2f} time units).")

# Create and run the simulation
simulation = NBodySimulation(
    particles,
    boundary,
    theta=theta,
    softening=softening,
    max_capacity=max_capacity,
    tree_rebuild_interval=rebuild_interval
)
start = time.time()
simulation.run(steps=int(t_final / dt), dt=dt, save_interval=save_interval, save_path='outputs/saves/')
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds.")   