from quadtree import QuadTree, Rectangle, Body

class Particle(Body):
    def __init__(self, x: float, y: float, mass: float, vx: float = 0.0, vy: float = 0.0):
        super().__init__(x, y)
        self.mass = mass
        self.vx = vx
        self.vy = vy
        
    
class BarnesHut(QuadTree):
    def __init__(self, boundary: Rectangle, max_capacity: int = 1, depth: int = 0):
        super().__init__(boundary, max_capacity, depth)
        self.total_mass = 0.0
        self.center_of_mass = (0.0, 0.0)
        
    def _create_child(self, boundary: Rectangle) -> QuadTree:
        return BarnesHut(boundary, self.max_capacity, self.depth + 1)
        
    def insert(self, body: Body) -> bool:
        if not isinstance(body, Particle):
            raise TypeError("BarnesHut can only insert Particle instances")

        particle = body

        if not self.boundary.contains(particle):
            return False
        
        # Update total mass and center of mass
        self.total_mass += particle.mass
        self.center_of_mass = (
            (self.center_of_mass[0] * (self.total_mass - particle.mass) + particle.x * particle.mass) / self.total_mass,
            (self.center_of_mass[1] * (self.total_mass - particle.mass) + particle.y * particle.mass) / self.total_mass
        )
        
        return super().insert(particle)
    
    def compute_force(self, particle: Particle, theta: float, softening: float = 1e-3) -> tuple[float, float]:
        if self.total_mass == 0:
            return (0.0, 0.0)

        if not self.divided:
            force_x, force_y = 0.0, 0.0
            for other in self.bodies:
                if other is particle:
                    continue
                dx = other.x - particle.x
                dy = other.y - particle.y
                dist_sq = dx * dx + dy * dy + softening * softening
                distance = dist_sq**0.5
                force_magnitude = (particle.mass * other.mass) / dist_sq
                force_x += force_magnitude * (dx / distance)
                force_y += force_magnitude * (dy / distance)
            return (force_x, force_y)
        
        dx = self.center_of_mass[0] - particle.x
        dy = self.center_of_mass[1] - particle.y
        dist_sq = dx * dx + dy * dy + softening * softening
        distance = dist_sq**0.5
        
        if distance == 0:
            return (0.0, 0.0)
        
        # Check if we can approximate the force using the center of mass
        if (self.boundary.right - self.boundary.left) / distance < theta:
            force_magnitude = (self.total_mass * particle.mass) / dist_sq
            force_direction = (dx / distance, dy / distance)
            return (force_magnitude * force_direction[0], force_magnitude * force_direction[1])
        else:
            force_x, force_y = 0.0, 0.0
            for child in self.children:
                if isinstance(child, BarnesHut):
                    fx, fy = child.compute_force(particle, theta, softening)
                    force_x += fx
                    force_y += fy
            return (force_x, force_y)   