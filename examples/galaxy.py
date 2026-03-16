from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from nbody.nbody_simulation import NBodySimulation, Particle, Rectangle
import matplotlib.pyplot as plt

def create_galaxy(num_stars: int, boundary: Rectangle, radius_bulge: float = 0.5, radius_halo: float = 15.0, radius_disk: float = 3.0, softening: float = 1e-9) -> list[Particle]:
    """
    Creates a galaxy with a specified number of stars, distributed in a disk shape.
    Parameters:
    - num_stars (int): The number of stars to create in the galaxy.
    - boundary (Rectangle): The boundary within which the galaxy will be created.
    - radius_bulge (float): The radius of the galaxy bulge.
    - radius_halo (float): The radius of the galaxy halo.
    - radius_disk (float): The radius of the galaxy disk.
    - softening (float): The softening parameter for force calculation.
    Returns:
    - list[Particle]: A list of Particle instances representing the stars in the galaxy.
    """
    # =======
    # Particle position generation
    # =======
    
    # Boundary
    center = ((boundary.left + boundary.right) / 2, (boundary.top + boundary.bottom) / 2)
    
    # Mass settings
    M_disk, M_halo, M_bulge = 1.0, 5.8, 0.2
    total_mass = M_disk + M_halo + M_bulge
    particle_mass = total_mass / num_stars
    
    # Particle distribution for disk, halo, and bulge
    num_disk = int(num_stars * M_disk / total_mass)
    num_halo = int(num_stars * M_halo / total_mass)
    num_bulge = num_stars - num_disk - num_halo
    
    # Bulge stars
    r_core = np.random.uniform(0, radius_bulge, num_bulge)
    theta_core = np.random.uniform(0, 2 * np.pi, num_bulge)
    # Halo stars
    r_halo = np.random.uniform(0, radius_halo, num_halo)
    theta_halo = np.random.uniform(0, 2 * np.pi, num_halo)
    # Disk stars
    r_disk = np.random.exponential(scale=radius_disk, size=num_disk)
    theta_disk = np.random.uniform(0, 2 * np.pi, num_disk)
    
    # Combine all and convert to Cartesian coordinates
    r = np.concatenate([r_core, r_halo, r_disk])
    theta = np.concatenate([theta_core, theta_halo, theta_disk])
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    # Enclosed mass estimate by rank in radius (monotonic and O(N log N))
    order = np.argsort(r)
    enclosed_mass = np.empty_like(r)
    enclosed_mass[order] = np.arange(num_stars, dtype=float) * particle_mass

    # Keep force/velocity estimates numerically well-behaved in the center
    softening_eff = max(softening, 5e-2)
    
    # =======
    # Particle velocity generation
    # =======
    # Simple circular velocity for disk stars, random velocities for halo and bulge
    all_particles = []
    for i in range(num_stars):
        # Circular speed from softened gravity: v^2 = M(<r) r^2 / (r^2 + eps^2)^(3/2)
        r_i = r[i]
        denom = (r_i * r_i + softening_eff * softening_eff) ** 1.5
        v_circ = np.sqrt(enclosed_mass[i] * r_i * r_i / denom) if denom > 0 else 0.0
        # Assign velocities
        if i < num_bulge + num_halo:
            vx = np.random.normal(0, v_circ * 0.25)
            vy = np.random.normal(0, v_circ * 0.25)
            alpha = 0.0
        else:
            v_rot = 0.9 * v_circ
            vx = -v_rot * np.sin(theta[i])
            vy = v_rot * np.cos(theta[i])
            # Add some random perturbation to disk star velocities
            vx += np.random.normal(0, v_rot * 0.03)
            vy += np.random.normal(0, v_rot * 0.03)
            alpha = 1.0
        all_particles.append(Particle(x[i], y[i], particle_mass, vx, vy, alpha=alpha))
    
    return all_particles


if __name__ == "__main__":
    # Simulation parameters
    num_stars = 1000
    boundary = Rectangle(0, 100, 100, 0)
    radius_bulge = 0.5
    radius_halo = 15.0
    radius_disk = 3.0
    softening = 1e-9
    max_capacity = 16
    theta = 0.9
    # Create galaxy and run simulation
    stars = create_galaxy(num_stars, boundary, radius_bulge, radius_halo, radius_disk, softening)
    simulation = NBodySimulation(
        stars,
        boundary,
        theta=theta,
        max_capacity=max_capacity,
    )
    simulation.run(steps=1000, dt=0.01, save_interval=-1, save_path='outputs/saves/')
    
    # # Plot initial galaxy
    # fig, ax = plt.subplots()
    # ax.scatter([star.x for star in stars], [star.y for star in stars], s=1, color=[(1, 0, 0, star.alpha) for star in stars], zorder=10)
    # ax.set_xlim(boundary.left, boundary.right)
    # ax.set_ylim(boundary.bottom, boundary.top)
    # ax.set_title("Initial Galaxy Configuration")
    # plt.show()