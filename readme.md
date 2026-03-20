# N-body Simulation with Barnes-Hut Algorithm
This is a simple N-body simulation implemented in Python using the Barnes-Hut algorithm for efficient force calculation. The simulation is going to be deployed on a web server using Streamlit, allowing users to interact with the simulation in real-time. 

https://github.com/user-attachments/assets/faefe1fe-f2e5-481c-909a-a3b0713a99b5
## Project Structure

- `nbody/`: reusable simulation engine and Cython extension
- `examples/`: runnable scenarios (easy to extend with new setups)
- `tools/`: utilities such as state animation/export
- `outputs/`: generated simulation data

## Quick Start

Build Cython extension:

```bash
python setup.py build_ext --inplace
```

Run examples:

```bash
python examples/spherical_collapse.py
python examples/galaxy.py
```

Animate saved states:

```bash
python tools/animate_states.py
```

See `examples/README.md` for guidance on adding more examples.

## Goals
- Gaining first experiences with Streamlit for web deployment.
- Implementing the Barnes-Hut algorithm for efficient N-body simulations.
- Visualizing the simulation results in an interactive web interface.

## Features
- [x] First tests with Streamlit for web deployment.
- [x] Implementation of the Barnes-Hut algorithm for N-body simulations.
- [ ] Interactive visualization of the simulation results.
- [ ] User controls for adjusting simulation parameters (e.g., number of bodies, time step, etc.).
- [x] Switch from Python to Cython for performance optimization 

## Barnes-Hut Algorithm
The Barnes-Hut algorithm is a method for efficiently simulating the gravitational interactions between a large number of bodies. It works by dividing the simulation space into a hierarchical tree structure (octree in 3D, quadtree in 2D) and approximating the forces from distant bodies using their center of mass. This reduces the computational complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$, making it feasible to simulate larger systems.

## Optimization with Cython
To further optimize the performance of the N-body simulation, we can switch from Python to Cython. Cython allows us to write C-like code in Python, which can be compiled to C for significantly faster execution. This is particularly beneficial for computationally intensive tasks like the Barnes-Hut algorithm, where the performance can be a bottleneck. 
By switching to Cython, I achieved a speedup of approximately 10x compared to the pure Python implementation, allowing for larger simulations and faster results.

## Sources
- [Barnes-Hut Algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [The Barnes-Hut Algorithm by Tom Ventimiglia & Kevin Wayne](https://www.arborjs.com/docs/barnes-hut) 
