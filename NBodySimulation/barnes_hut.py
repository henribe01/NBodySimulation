from NBodySimulation.quadtree import QuadTree, Rectangle, Body
import numba

class Particle(Body):
    def __init__(self, x: float, y: float, mass: float, vx: float = 0.0, vy: float = 0.0, color: str = 'r', alpha: float = 1.0) -> None:
        super().__init__(x, y)
        if mass <= 0:
            raise ValueError("Particle mass must be greater than zero")
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.vx = vx
        self.vy = vy
        self.color = color
        self.alpha = alpha
        
    
class BarnesHut(QuadTree):
    def __init__(self, boundary: Rectangle, max_capacity: int = 1, depth: int = 0, max_depth: int = 20, min_cell_size: float = 1e-6):
        super().__init__(boundary, max_capacity, depth, max_depth, min_cell_size)
        self.total_mass = 0.0
        self.center_of_mass = (0.0, 0.0)
        
    def _create_child(self, boundary: Rectangle) -> QuadTree:
        return BarnesHut(
            boundary,
            self.max_capacity,
            self.depth + 1,
            self.max_depth,
            self.min_cell_size,
        )
        
    def insert(self, body: Body) -> bool:
        if not isinstance(body, Particle):
            raise TypeError("BarnesHut can only insert Particle instances")

        particle = body

        if not self.boundary.contains(particle):
            return False
        
        # Update total mass and center of mass using numerically stable formula
        old_total = self.total_mass
        self.total_mass += particle.mass
        if old_total > 0:
            # Incremental update to center of mass
            self.center_of_mass = (
                (self.center_of_mass[0] * old_total + particle.x * particle.mass) / self.total_mass,
                (self.center_of_mass[1] * old_total + particle.y * particle.mass) / self.total_mass
            )
        else:
            self.center_of_mass = (particle.x, particle.y)
        
        return super().insert(particle)
    
    def compute_force(self, particle: Particle, theta: float, softening: float = 1e-3) -> tuple[float, float]:
        if self.total_mass == 0:
            return (0.0, 0.0)

        theta_sq = theta * theta

        if not self.divided:
            force_x, force_y = 0.0, 0.0
            eps_sq = softening * softening
            for other in self.bodies:
                if other is particle:
                    continue
                dx = other.x - particle.x
                dy = other.y - particle.y
                dist_sq = dx * dx + dy * dy + eps_sq
                # Avoid sqrt: force_x += force_magnitude * (dx / distance)
                # is equivalent to: force_x += (mass1 * mass2 / dist_sq) * (dx / sqrt(dist_sq))
                # = force_x += (mass1 * mass2 * dx) / (dist_sq * sqrt(dist_sq))
                # = force_x += (mass1 * mass2 * dx) / dist_sq^1.5
                inv_dist_32 = dist_sq ** -1.5  # Compute once, use twice
                force_x += (particle.mass * other.mass * dx) * inv_dist_32
                force_y += (particle.mass * other.mass * dy) * inv_dist_32
            return (force_x, force_y)
        
        dx = self.center_of_mass[0] - particle.x
        dy = self.center_of_mass[1] - particle.y
        eps_sq = softening * softening
        dist_sq = dx * dx + dy * dy + eps_sq
        
        if dist_sq == 0:
            return (0.0, 0.0)
        
        # Check if we can approximate the force using the center of mass
        size = self.boundary.right - self.boundary.left
        if (size * size) < (theta_sq * dist_sq):
            force_magnitude = (self.total_mass * particle.mass) / dist_sq
            force_direction = (dx / distance, dy / distance)
            return (force_magnitude * force_direction[0], force_magnitude * force_direction[1])
        else:
            force_x, force_y = 0.0, 0.0
            for child in self.children:
                if isinstance(child, BarnesHut) and child.total_mass > 0:
                    fx, fy = child.compute_force(particle, theta, softening)
                    force_x += fx
                    force_y += fy
            return (force_x, force_y)   