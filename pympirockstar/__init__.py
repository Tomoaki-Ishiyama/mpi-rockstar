"""Python interface for the MPI-Rockstar library."""

from ._rockstar import run, RockstarError


def run_config(config: str) -> int:
    """Run ``mpi-rockstar`` using the given configuration file.

    Parameters
    ----------
    config:
        Path to a Rockstar configuration file.

    Returns
    -------
    int
        Exit code returned by :func:`run`.
    """

    return run(["mpi-rockstar", "-c", config])


__all__ = ["run", "run_config", "RockstarError"]
