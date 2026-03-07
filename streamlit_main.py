import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Settings for animation
N = 100
t_final = 10
dt = 0.1
t = np.arange(0, t_final, dt)
pos_min, pos_max = -5, 5

# Randomly initial positions
positions = np.random.uniform(pos_min, pos_max, (N, 2))

# Randomly move particles
def update(frame):
    global positions
    # Random walk
    step = np.random.normal(0, 0.1, (N, 2))
    positions += step
    # Periodic boundary conditions
    positions = np.mod(positions - pos_min, pos_max - pos_min) + pos_min
    scat.set_offsets(positions)
    return scat,

# Create figure and scatter plot
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1], c='r', s=5)
ax.set_xlim(pos_min, pos_max)
ax.set_ylim(pos_min, pos_max)
ax.set_title('Random Walk of Particles')
# Create animation
ani = FuncAnimation(fig, update, frames=len(t), blit=True)
ani.save('random_walk.gif', writer='Pillow', fps=30)

# Streamlit app
st.title("Random Walk Animation")
st.image('random_walk.gif', caption='Random Walk of Particles', use_column_width=True)