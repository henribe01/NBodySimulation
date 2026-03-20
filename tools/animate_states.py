import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

FPS = 30
BG_COLOR = '#ffffff'
AX_COLOR = '#ffffff'
GRID_COLOR = '#d0d7de'
TEXT_COLOR = '#1f2328'


def _resolve_default_paths(settings_path: str, states_path: str) -> tuple[str, str]:
    if os.path.exists(settings_path) and os.path.exists(states_path):
        return settings_path, states_path

    legacy_settings = 'saves/settings.npz'
    legacy_states = 'saves/'
    if os.path.exists(legacy_settings) and os.path.exists(legacy_states):
        return legacy_settings, legacy_states

    return settings_path, states_path

def load_simulation_settings(path: str = 'outputs/saves/settings.npz') -> dict:
    """
    Loads the simulation settings from a file.
    
    Parameters:
    - path (str): The file path where the settings are stored.
    
    Returns:
    - dict: A dictionary containing the simulation settings.
    """
    data = dict(np.load(path, allow_pickle=True))
    return data


def load_simulation_states(path: str = 'outputs/saves/') -> list:
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
        max_velocity = max(max_velocity, np.percentile(velocities, 99.5))
    return (0, max_velocity)


def _get_plot_limits(settings: dict, states: list) -> tuple[float, float, float, float]:
    boundary = settings.get('boundary')
    if boundary is not None and len(boundary) >= 4:
        xmin, xmax = float(boundary[0]), float(boundary[1])
        ymin, ymax = float(boundary[3]), float(boundary[2])
        return xmin, xmax, ymin, ymax

    all_x = np.concatenate([state['x'] for state in states])
    all_y = np.concatenate([state['y'] for state in states])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

    x_pad = max(0.05 * (x_max - x_min), 1e-6)
    y_pad = max(0.05 * (y_max - y_min), 1e-6)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad
    
def animate_simulation(states: list, settings: dict, save: bool = False) -> None:
    """
    Creates an animation of the N-body simulation using the loaded states and settings.
    
    Parameters:
    - states (list): A list of dictionaries containing the simulation states at each step.
    - settings (dict): A dictionary containing the simulation settings.
    """
    vmin, vmax = get_color_scale(states)
    x_min, x_max, y_min, y_max = _get_plot_limits(settings, states)
    dt = float(settings.get('dt', 1.0))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_COLOR, constrained_layout=True)
    ax.set_facecolor(AX_COLOR)
    scat = ax.scatter([], [], s=2, c=[], cmap='plasma', vmin=vmin, vmax=vmax, zorder=10, marker='o', edgecolors='none')
    cbar = plt.colorbar(scat, ax=ax, pad=0.02)
    cbar.set_label('Speed', color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    frame_label = ax.text(
        0.02,
        0.98,
        '',
        transform=ax.transAxes,
        ha='left',
        va='top',
        color=TEXT_COLOR,
        fontsize=10,
        bbox=dict(facecolor='#ffffff', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.25')
    )

    def init():
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.set_title('N-Body Simulation', color=TEXT_COLOR, fontsize=14, pad=10)
        ax.set_xlabel('X Position', color=TEXT_COLOR)
        ax.set_ylabel('Y Position', color=TEXT_COLOR)
        ax.grid(True, zorder=0, alpha=0.5, color=GRID_COLOR, linewidth=0.7)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        return scat, frame_label

    def update(frame):
        scat.set_offsets(np.c_[states[frame]['x'], states[frame]['y']])
        speeds = np.sqrt(states[frame]['vx']**2 + states[frame]['vy']**2)
        scat.set_array(speeds)
        frame_label.set_text(f'Frame: {frame + 1}/{len(states)}\nTime: {frame * dt:.2f}')
        return scat, frame_label

    anim = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True, interval=1000/FPS)
    if save:
        anim.save('nbody_simulation.mp4', writer='ffmpeg', fps=FPS, dpi=140, extra_args=['-vcodec', 'libx264', 
                                                                                         '-crf', '21',
                                                                                         '-preset', 'slow',
                                                                                         '-pix_fmt', 'yuv420p',
                                                                                         '-movflags', '+faststart'])
    else:
        plt.show()
    
if __name__ == "__main__":
    settings_path, states_path = _resolve_default_paths('outputs/saves/settings.npz', 'outputs/saves/')
    settings = load_simulation_settings(settings_path)
    states = load_simulation_states(states_path)
    animate_simulation(states, settings, save=True)
