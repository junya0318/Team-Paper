# dr-music-improved (course project scaffold)

This folder is a **refactor scaffold** based on the public repository:
- DA-MUSIC / DR-MUSIC_ICASSP23 (Deep Root-MUSIC; DoA estimation)

What I added (typical “course topics”):
- `pyproject.toml` for packaging (uv/PEP 621 compatible)
- minimal `src/` layout
- type hints + docstrings for key functions
- `pytest` unit tests for numerical helper functions
- reproducibility helper (`set_seed`)

## Quick start (uv)

```bash
uv venv
uv pip install -e ".[dev]"
pytest
python -m dr_music.cli --seed 0 --tau 2
```

## Notes

- `legacy/` contains the original files we extracted for reference.
- This is **not** a full reproduction of the original training pipeline yet;
  it is the “minimum refactor” you can present in the final report.
