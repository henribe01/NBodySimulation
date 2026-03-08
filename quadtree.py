import numpy as np

class Body:
    def __init__(self, x: float, y: float) -> None:
        """
        Initializes a Body object with a position (x, y).
        
        Parameters:
        - x: The x-coordinate of the body.
        - y: The y-coordinate of the body.
        """
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, left: float, right: float, top: float, bottom: float) -> None:
        """
        Initializes a Rectangle object defined by its left, right, top, and bottom edges.
        
        Parameters:
        - left: The x-coordinate of the left edge of the rectangle.
        - right: The x-coordinate of the right edge of the rectangle.
        - top: The y-coordinate of the top edge of the rectangle.
        - bottom: The y-coordinate of the bottom edge of the rectangle.
        """
        self.left = min(left, right)
        self.right = max(left, right)
        self.top = max(top, bottom)
        self.bottom = min(top, bottom)
        
    def contains(self, body: Body) -> bool:
        """
        Checks if the rectangle contains a given body.
        
        Parameters:
        - body: The Body object to check for containment.
        
        Returns:
        - True if the rectangle contains the body, False otherwise.
        """
        return (self.left <= body.x < self.right) and (self.bottom <= body.y < self.top)
    
    def intersects(self, other: Rectangle) -> bool:
        """
        Checks if this rectangle intersects with another rectangle.
        
        Parameters:
        - other: Another Rectangle object to check for intersection.
        
        Returns:
        - True if the rectangles intersect, False otherwise.
        """
        return not (other.left >= self.right or 
                    other.right <= self.left or 
                    other.top <= self.bottom or 
                    other.bottom >= self.top)
        
    def draw(self, ax, **kwargs) -> None:
        """
        Draws the rectangle on a given Matplotlib Axes.
        
        Parameters:
        - ax: The Matplotlib Axes to draw the rectangle on.
        - kwargs: Additional keyword arguments for the rectangle patch (e.g., edgecolor, facecolor).
        """
        import matplotlib.patches as patches

        rect = patches.Rectangle((self.left, self.bottom), self.right - self.left, self.top - self.bottom, **kwargs)
        ax.add_patch(rect)
    

class QuadTree:
    def __init__(self, boundary: Rectangle, max_capacity: int = 1, depth: int = 0, max_depth: int = 20, min_cell_size: float = 1e-6) -> None:
        """
        Initializes a QuadTree node.
        
        Parameters:
        - boundary: A Rectangle object defining the rectangular boundary of the node.
        - max_capacity: The maximum number of bodies that can be stored in a leaf node before it subdivides.
        - depth: The depth of the node in the quadtree.
        """
        assert isinstance(boundary, Rectangle), "Boundary must be a Rectangle object"
        self.boundary = boundary
        self.max_capacity = max_capacity
        self.bodies = []
        self.depth = depth
        self.max_depth = max_depth
        self.min_cell_size = min_cell_size
        self.children: list[QuadTree | None] = [None, None, None, None]  # NW, NE, SW, SE
        self.divided = False
        
    def insert(self, body: Body) -> bool:
        """
        Inserts a body into the quadtree.
        
        Parameters:
        - body: The Body object to be inserted.
        
        Returns:
        - True if the body was successfully inserted, False otherwise.
        """
        # Check if the body is within the boundary of this node
        if not self.boundary.contains(body):
            return False
        
        # If the node has space and is not divided, add the body here
        if len(self.bodies) < self.max_capacity and not self.divided:
            self.bodies.append(body)
            return True

        # Stop subdividing when hitting depth/cell-size limits
        if not self.divided and not self._can_subdivide():
            self.bodies.append(body)
            return True
        
        # If the node is full and not divided, subdivide it and redistribute the bodies
        if not self.divided:
            subdivided = self.subdivide()
            if not subdivided:
                self.bodies.append(body)
                return True
        for child in self.children:
            if child is not None and child.insert(body):
                return True
        
        return False
    
    def subdivide(self) -> bool:
        """
        Subdivides the current node into four child nodes (NW, NE, SW, SE).
        """
        if not self._can_subdivide():
            return False

        mid_x = (self.boundary.left + self.boundary.right) / 2
        mid_y = (self.boundary.top + self.boundary.bottom) / 2
        
        nw_boundary = Rectangle(self.boundary.left, mid_x, self.boundary.top, mid_y)
        ne_boundary = Rectangle(mid_x, self.boundary.right, self.boundary.top, mid_y)
        sw_boundary = Rectangle(self.boundary.left, mid_x, mid_y, self.boundary.bottom)
        se_boundary = Rectangle(mid_x, self.boundary.right, mid_y, self.boundary.bottom)
        
        self.children[0] = self._create_child(nw_boundary)  # NW
        self.children[1] = self._create_child(ne_boundary)  # NE
        self.children[2] = self._create_child(sw_boundary)  # SW
        self.children[3] = self._create_child(se_boundary)  # SE
        
        # Redistribute existing bodies into the appropriate child nodes
        for body in self.bodies:
            for child in self.children:
                if child is not None and child.insert(body):
                    break
        
        # Clear the bodies from the current node since they are now in child nodes
        self.bodies.clear()
        self.divided = True
        return True
        
    def query(self, boundary: Rectangle, found_points: list[Body]) -> None:
        """
        Queries the quadtree for bodies within a given rectangular boundary.
        
        Parameters:
        - boundary: A Rectangle object defining the query area.
        - found_points: A list to store the bodies found within the query area.
        """
        # Check if the query boundary intersects with this node's boundary
        if not self.boundary.intersects(boundary):
            return
        
        # If this node is a leaf, check if the bodies are within the query boundary
        if not self.divided:
            for body in self.bodies:
                if boundary.contains(body):
                    found_points.append(body)
        else:
            # If this node is divided, query the child nodes
            for child in self.children:
                if child is not None:
                    child.query(boundary, found_points)
                    
    def draw(self, ax, **kwargs) -> None:
        """
        Draws the quadtree boundaries on a given Matplotlib Axes.
        
        Parameters:
        - ax: The Matplotlib Axes to draw the quadtree on.
        - kwargs: Additional keyword arguments for the rectangle patch (e.g., edgecolor, facecolor).
        """
        self.boundary.draw(ax, **kwargs)
        if self.divided:
            for child in self.children:
                if child is not None:
                    child.draw(ax, **kwargs)
                    
    def _create_child(self, boundary: Rectangle) -> QuadTree:
        """
        Creates a child QuadTree node with the given boundary.
        
        Parameters:
        - boundary: A Rectangle object defining the boundary of the child node.
        
        Returns:
        - A new QuadTree instance representing the child node.
        """
        return QuadTree(
            boundary,
            self.max_capacity,
            self.depth + 1,
            self.max_depth,
            self.min_cell_size,
        )

    def _can_subdivide(self) -> bool:
        if self.depth >= self.max_depth:
            return False

        width = self.boundary.right - self.boundary.left
        height = self.boundary.top - self.boundary.bottom
        return width > self.min_cell_size and height > self.min_cell_size
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage of the QuadTree
    
    # Function to generate random bodies (gaussian distribution around the center)
    def generate_random_bodies(n: int, pos_max: float, std: float) -> list[Body]:
        center = pos_max / 2
        bodies = []
        for _ in range(n):
            x = np.random.normal(center, std)
            y = np.random.normal(center, std)
            bodies.append(Body(x, y))
        return bodies

    N = 5000
    pos_max = 100
    std = pos_max / 6
    boundary = Rectangle(0, pos_max, pos_max, 0)
    quadtree = QuadTree(boundary, max_capacity=4)
    bodies = generate_random_bodies(N, pos_max, std)
    for body in bodies:
        quadtree.insert(body)
        
    # Query the quadtree for bodies within a specific area
    found_bodies = []
    quadtree.query(boundary, found_bodies)
    print(f"Found {len(found_bodies)} bodies in the quadtree.")
    
    # Plot the bodies and the quadtree boundaries
    fig, ax = plt.subplots()
    ax.scatter([body.x for body in found_bodies], [body.y for body in found_bodies], s=2.5, color='r', zorder=10)
    quadtree.draw(ax, edgecolor='blue', facecolor='none', linewidth=0.5, zorder=11)
    ax.set_xlim(0, pos_max)
    ax.set_ylim(0, pos_max)
    ax.set_aspect('equal')
    plt.title("Bodies in QuadTree")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()