from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from nbody.nbody_simulation import NBodySimulation, Particle, Rectangle
import matplotlib.pyplot as plt

def generate_particles(num_stars: int, center: tuple[float, float] = (0.0, 0.0), initial_velocity: tuple[float, float] = (0.0, 0.0),
                       min_radius: float = 0.2, max_radius: float = 2.0, mass_bulge: float = 0.8, mass_disk: float = 0.2, add_noise: bool = False) -> list[Particle]:
    """
    Generate particles in a disk distribution with a central bulge.
    """
    # Bulge
    x_bulge = np.array([center[0]])  # Single particle at the center for the bulge
    y_bulge = np.array([center[1]])
    vx_bulge = np.array([initial_velocity[0]])  # Initial velocity for the bulge
    vy_bulge = np.array([initial_velocity[1]])
    mass_bulge_particle = np.array([mass_bulge])  # Total mass of the bulge
    
    # Disk
    # Salpeter-like mass distribution for the disk particles
    # m_disk_particles = np.ones(num_stars) * (mass_disk / num_stars)  # Mass of each disk particle
    alpha = 2.35  # Salpeter slope
    m_min = 0.1
    m_max = 100.0
    u = np.random.uniform(0, 1, num_stars)
    power = 1 - alpha
    masses_raw = (m_min**power + u * (m_max**power - m_min**power))**(1/power)
    m_disk_particles = masses_raw * (mass_disk / np.sum(masses_raw))
    # Generate positions and velocities for the disk particles
    angles = np.random.uniform(0, 2 * np.pi, num_stars)
    u = np.random.uniform(0, 1, num_stars)
    radii = np.sqrt(u * (max_radius**2 - min_radius**2) + min_radius**2)  # Uniform distribution in area
    x_disk = radii * np.cos(angles) + center[0]
    y_disk = radii * np.sin(angles) + center[1]
    # Circular velocities for the disk particles
    fractions_of_disk_inside_r = (radii**2 - min_radius**2) / (max_radius**2 - min_radius**2)
    M_eclosed = mass_bulge_particle + (m_disk_particles * fractions_of_disk_inside_r)
    v_mag = np.sqrt(M_eclosed / radii)  # Circular velocity
    vx_disk = -v_mag * np.sin(angles)
    vy_disk = v_mag * np.cos(angles) 
    # Add noise to velocities if specified
    if add_noise:
        sigma = 0.1 * v_mag 
        vx_disk += np.random.normal(0, sigma, num_stars)
        vy_disk += np.random.normal(0, sigma, num_stars)
    # Shift disk velocities by the initial velocity
    vx_disk += initial_velocity[0]
    vy_disk += initial_velocity[1]
    
    # Combine bulge and disk particles
    x = np.concatenate((x_bulge, x_disk))
    y = np.concatenate((y_bulge, y_disk))
    vx = np.concatenate((vx_bulge, vx_disk))
    vy = np.concatenate((vy_bulge, vy_disk))
    masses = np.concatenate((mass_bulge_particle, m_disk_particles))
    particles = [Particle(x[i], y[i], masses[i], vx[i], vy[i]) for i in range(num_stars + 1)]
    return particles

if __name__ == "__main__":
    # Simulation settings
    num_stars = 10000
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
    animation_time = 60  # Total time for the animation in seconds
    save_interval = max(1, int((t_final / (fps * animation_time)) / dt))  # Save every few steps to achieve desired fps and animation length
    print(f"Saving output every {save_interval} steps (every {save_interval * dt:.2f} time units).")

    # Generate particles
    particles = generate_particles(num_stars, add_noise=True)
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