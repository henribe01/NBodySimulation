import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

FPS = 30

def load_simulation_settings(path: str = 'saves/settings.npz') -> dict:
    """
    Loads the simulation settings from a file.
    
    Parameters:
    - path (str): The file path where the settings are stored.
    
    Returns:
    - dict: A dictionary containing the simulation settings.
    """
    data = dict(np.load(path, allow_pickle=True))
    return data


def load_simulation_states(path: str = 'saves/') -> list:
    """
    Loads the simulation states from a directory.
    
    Parameters:
    - path (str): The directory path where the simulation states are stored.
    
    Returns:
    - list: A list of dictionaries, each containing the state of the simulation at a given step.
    """
    states = []
    for file in sorted(os.listdir(path)):
        if file.startswith('state_') and file.endswith('.npz'):
            data = dict(np.load(os.path.join(path, file)))
            states.append({
                'x': data['x'],
                'y': data['y'],
                'vx': data['vx'],
                'vy': data['vy']
            })
    print(f"Loaded {len(states)} states from {path}")
    return states

def get_color_scale(states: list) -> tuple:
    """
    Determines the color scale for the animation based on the velocity magnitudes in the states.
    
    Parameters:
    - states (list): A list of dictionaries containing the simulation states at each step.
    
    Returns:
    - tuple: A tuple containing the minimum and maximum velocity magnitudes for color scaling.
    """
    max_velocity = 0
    for state in states:
        velocities = np.sqrt(state['vx']**2 + state['vy']**2)
        max_velocity = max(max_velocity, np.max(velocities))
    return (0, max_velocity)
    
def animate_simulation(states: list, settings: dict) -> None:
    """
    Creates an animation of the N-body simulation using the loaded states and settings.
    
    Parameters:
    - states (list): A list of dictionaries containing the simulation states at each step.
    - settings (dict): A dictionary containing the simulation settings.
    """
    vmin, vmax = get_color_scale(states)

    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=1, c=[], cmap='plasma', vmin=vmin, vmax=vmax, zorder=10)
    plt.colorbar(scat, ax=ax, label='Speed')

    def init():
        ax.set_xlim(settings['boundary'][0], settings['boundary'][1])
        ax.set_ylim(settings['boundary'][3], settings['boundary'][2])
        ax.set_aspect('equal')
        ax.set_title('N-Body Simulation', color='white')
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.grid(True, zorder=0, alpha=0.3)
        return scat,

    def update(frame):
        scat.set_offsets(np.c_[states[frame]['x'], states[frame]['y']])
        speeds = np.sqrt(states[frame]['vx']**2 + states[frame]['vy']**2)
        scat.set_array(speeds)
        return scat,

    anim = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True, interval=1000/FPS)
    # plt.show()
    anim.save('nbody_simulation.mp4', writer='ffmpeg', fps=FPS)
    
if __name__ == "__main__":
    settings = load_simulation_settings()
    states = load_simulation_states()
    animate_simulation(states, settings)