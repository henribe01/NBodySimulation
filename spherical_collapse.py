from nbody_simulation import NBodySimulation, Particle, Rectangle
import numpy as np
import time

def generate_particles(num_stars: int, radius: float = 1.0, total_mass: float = 1.0) -> list[Particle]:
    """
    Generate particles in a spherical distribution without initial velocities.
    """
    # Total mass is normalized to 1 (N-Body units)
    masses = np.ones(num_stars) * (total_mass / num_stars)
    
    # Generate random positions in a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_stars)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_stars))  # sqrt for uniform distribution in volume
    
    # Convert spherical to Cartesian coordinates
    x = radii * np.cos(phi)
    y = radii * np.sin(phi)
    
    # Cold collapse: initial velocities are zero
    vx = np.zeros(num_stars)
    vy = np.zeros(num_stars)

    particles = [Particle(x[i], y[i], masses[i], vx[i], vy[i]) for i in range(num_stars)]
    return particles

# Simulation settings
num_stars = 5000
softening = 0.05
dt = 0.01
boundary = Rectangle(-50, 50, 50, -50)
max_capacity = 4
theta = 0.8
rebuild_interval = 1
radius = 10.0
total_mass = 5.0

# Calculate timescales
t_relax = np.sqrt(radius**3 / total_mass)  # Dynamical time for the system
t_final = 5 * t_relax  # Run for several dynamical times to see the collapse
t_final = np.ceil(t_final)
print(f"Estimated relaxation time: {t_relax:.2f} time units")
print(f"Running simulation for {t_final:.2f} time units with dt={dt:.2f} time units.")

# Save interval for output
fps = 30  # Desired frames per second for output
animation_time = 30  # Total time for the animation in seconds
save_interval = max(1, int((t_final / (fps * animation_time)) / dt))  # Save every few steps to achieve desired fps and animation length
print(f"Saving output every {save_interval} steps (every {save_interval * dt:.2f} time units).")

# Generate particles
particles = generate_particles(num_stars, radius=radius, total_mass=total_mass)
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
simulation.run(steps=int(t_final / dt), dt=dt, save_interval=save_interval, save_path='saves/')
end = time.time()
print(f"Simulation completed in {end - start:.2f} seconds.")