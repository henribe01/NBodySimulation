# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport sqrt

cdef class Particle:
    cdef public double x, y
    cdef public double vx, vy
    cdef public double mass

    cdef public double inv_mass

    def __init__(self, double x, double y, double mass, double vx=0.0, double vy=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        if mass > 0:
            self.inv_mass = 1.0 / mass
        else:
            self.inv_mass = 0.0

cdef class Rect:
    cdef public double x_min, x_max, y_min, y_max

    def __init__(self, double left, double right, double top, double bottom):
        self.x_min = left if left < right else right
        self.x_max = right if right > left else left
        self.y_min = bottom if bottom < top else top
        self.y_max = top if top > bottom else bottom
    
    cpdef bint contains(self, Particle p):
        return (self.x_min <= p.x < self.x_max) and (self.y_min <= p.y < self.y_max)
    
    cdef bint intersects(self, Rect other):
        return not (self.x_max <= other.x_min or self.x_min >= other.x_max or
                    self.y_max <= other.y_min or self.y_min >= other.y_max)
    
    def draw(self, ax, **kwargs):
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x_min, self.y_min), self.x_max - self.x_min, self.y_max - self.y_min, **kwargs)
        ax.add_patch(rect)

cdef class BarnesHut:
    cdef Rect boundary
    cdef int capacity
    cdef int depth
    cdef bool divided
    cdef int count
    cdef list particles
    cdef double total_mass
    cdef double center_of_mass_x, center_of_mass_y
    cdef BarnesHut nw, ne, sw, se

    def __init__(self, Rect boundary, int max_capacity=1, int depth=0):
        self.boundary = boundary
        self.capacity = max_capacity
        self.count = 0
        self.depth = depth
        self.divided = False
        self.total_mass = 0.0
        self.center_of_mass_x = 0.0
        self.center_of_mass_y = 0.0
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        self.particles = [None] * self.capacity
    
    def __dealloc__(self):
        pass

    cpdef bint insert(self, Particle p):
        # Check if the particle is within the boundary
        if not self.boundary.contains(p):
            return False
        
        # Update center of mass and total mass
        self.total_mass += p.mass
        self.center_of_mass_x = (self.center_of_mass_x * (self.total_mass - p.mass) + p.x * p.mass) / self.total_mass if self.total_mass > 0 else 0
        self.center_of_mass_y = (self.center_of_mass_y * (self.total_mass - p.mass) + p.y * p.mass) / self.total_mass if self.total_mass > 0 else 0

        # If this node is not divided and has capacity, insert the particle
        if not self.divided and self.count < self.capacity:
            self.particles[self.count] = p
            self.count += 1
            return True
        
        # If node is full and not divided, subdivide it and put particle into the right one
        if not self.divided:
            self.subdivide()
        if self.nw.insert(p): return True
        if self.ne.insert(p): return True
        if self.sw.insert(p): return True
        if self.se.insert(p): return True
        return False

    cdef void subdivide(self):
        cdef double mid_x = (self.boundary.x_min + self.boundary.x_max) / 2
        cdef double mid_y = (self.boundary.y_min + self.boundary.y_max) / 2
        cdef int i
        cdef Particle p

        cdef Rect nw_boundary = Rect(self.boundary.x_min, mid_x, mid_y, self.boundary.y_max)
        cdef Rect ne_boundary = Rect(mid_x, self.boundary.x_max, mid_y, self.boundary.y_max)
        cdef Rect sw_boundary = Rect(self.boundary.x_min, mid_x, self.boundary.y_min, mid_y)
        cdef Rect se_boundary = Rect(mid_x, self.boundary.x_max, self.boundary.y_min, mid_y)

        self.nw = BarnesHut(nw_boundary, self.capacity, self.depth + 1)
        self.ne = BarnesHut(ne_boundary, self.capacity, self.depth + 1)
        self.sw = BarnesHut(sw_boundary, self.capacity, self.depth + 1)
        self.se = BarnesHut(se_boundary, self.capacity, self.depth + 1)
        self.divided = True

        # Redistribute existing particles into the appropriate quadrants
        for i in range(self.count):
            p = <Particle>self.particles[i]
            if self.nw.insert(p):
                continue
            if self.ne.insert(p):
                continue
            if self.sw.insert(p):
                continue
            if self.se.insert(p):
                continue
        self.count = 0  # Clear the count as particles are now in child nodes

    cpdef query(self, Rect range, list result):
        cdef int i
        cdef Particle p
        if not self.boundary.intersects(range):
            return
        
        for i in range(self.count):
            p = <Particle>self.particles[i]
            if range.contains(p):
                result.append(p)
        
        if self.divided:
            self.nw.query(range, result)
            self.ne.query(range, result)
            self.sw.query(range, result)
            self.se.query(range, result)

    cdef double distance(self, Particle p1, Particle p2):
        cdef double dx = p1.x - p2.x
        cdef double dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)

    cpdef tuple compute_force(self, Particle p, double theta, double softening):
        # Python wrapper to initialize and return the force as a tuple
        cdef double fx = 0.0
        cdef double fy = 0.0
        cdef double theta_sq = theta * theta
        cdef double softening_sq = softening * softening

        self._compute_force(p, theta_sq, softening_sq, &fx, &fy)
        return fx, fy

    cdef void _compute_force(self, Particle p, double theta_sq, double softening_sq, double* fx, double* fy):
        # Pure C implementation of the force calculation
        cdef int i
        cdef Particle p_other
        cdef double dx_p, dy_p, dist_sq, distance, force_mag
        cdef double dx, dy, size

        # Check if node is empty
        if self.total_mass == 0:
            return
        
        # If this node is a leaf, compute force directly
        if not self.divided:
            for i in range(self.count):
                p_other = <Particle>self.particles[i]

                # Check if it's the same particle
                if p_other is p:
                    continue
                
                dx_p = p_other.x - p.x
                dy_p = p_other.y - p.y
                dist_sq = dx_p * dx_p + dy_p * dy_p + softening_sq
                distance = sqrt(dist_sq)
                force_mag = (p_other.mass * p.inv_mass) / dist_sq

                # Accumulate forces
                fx[0] += force_mag * dx_p / distance
                fy[0] += force_mag * dy_p / distance
            return
        
        # If this is an internal node, check if we can approximate it
        dx = self.center_of_mass_x - p.x
        dy = self.center_of_mass_y - p.y
        dist_sq = dx * dx + dy * dy + softening_sq
        distance = sqrt(dist_sq)

        if distance == 0:
            return
        
        size = self.boundary.x_max - self.boundary.x_min
        if (size * size) < (theta_sq * dist_sq):
            # Treat this node as a single body
            force_mag = (self.total_mass * p.inv_mass) / dist_sq
            fx[0] += force_mag * dx / distance
            fy[0] += force_mag * dy / distance
        else:
            # Otherwise, we need to go deeper into the tree
            if self.nw is not None:
                self.nw._compute_force(p, theta_sq, softening_sq, fx, fy)
            if self.ne is not None:
                self.ne._compute_force(p, theta_sq, softening_sq, fx, fy)
            if self.sw is not None:
                self.sw._compute_force(p, theta_sq, softening_sq, fx, fy)
            if self.se is not None:
                self.se._compute_force(p, theta_sq, softening_sq, fx, fy)


Rectangle = Rect


