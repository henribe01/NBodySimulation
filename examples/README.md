# Examples

Each file in this folder is a runnable scenario using the shared simulation engine in `nbody/`.

## Run examples

From project root:

```bash
python examples/spherical_collapse.py
python examples/galaxy.py
```

## Output location

Examples save simulation snapshots to:

- `outputs/saves/`

The animation tool reads from `outputs/saves/` by default and falls back to `saves/` for older runs:

```bash
python tools/animate_states.py
```

## Adding a new example

1. Create a new file in this folder, e.g. `examples/cluster_merge.py`.
2. Import from the package:

```python
from nbody.nbody_simulation import NBodySimulation, Particle, Rectangle
```

3. Save output to `outputs/saves/` (or a subfolder under `outputs/`).
4. Add your new command to this README.
