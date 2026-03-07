"""
Reader for AbacusSummit halo catalogues.

AbacusSummit is a suite of large cosmological N-body simulations run with the
Abacus code. Halo catalogues are produced by the CompaSO on-the-fly halo finder
and stored as ASDF files. This module wraps the ``abacusutils`` package to
extract the minimal set of halo properties required by ``jax-hod``.

Reference: Maksimova et al. 2021 (https://arxiv.org/abs/2110.11398)
"""

from pathlib import Path

import numpy as np


# Simulation name templates for the AbacusSummit suite.
# sim_type -> string prefix as used in the directory names.
_SIM_TYPES = {
    'base':   'AbacusSummit_base',    # 2000 Mpc/h, 6912^3 particles
    'small':  'AbacusSummit_small',   # 500  Mpc/h, 1728^3 particles
    'large':  'AbacusSummit_large',   # 2000 Mpc/h, 10240^3 particles (few sims)
    'huge':   'AbacusSummit_huge',    # 7500 Mpc/h, 8640^3 particles
    'highres':'AbacusSummit_highres', # 1000 Mpc/h, 6912^3 particles
}


def load_abacus_halos(
    sim_dir,
    cosmology,
    phase,
    redshift,
    sim_type='base',
    min_mass=None,
    cleaned=True,
    fields=None,
):
    """
    Load halo positions, masses, radii and velocities from an AbacusSummit
    simulation snapshot, ready to be passed directly to ``populate()``.

    Parameters
    ----------
    sim_dir : str or Path
        Path to the directory that contains the AbacusSummit simulations,
        e.g. ``/global/cfs/cdirs/desi/cosmosim/Abacus``.
    cosmology : str
        Cosmology tag, e.g. ``'c000'`` (Planck 2018 LCDM) or ``'c001'``.
    phase : str
        Phase tag, e.g. ``'ph000'``.  For secondary cosmologies that have
        only a single phase, use ``'ph000'``.
    redshift : float
        Snapshot redshift (e.g. ``0.5``). Must match one of the available
        AbacusSummit output redshifts.
    sim_type : str, optional
        Simulation volume/resolution type.  One of ``'base'`` (default),
        ``'small'``, ``'large'``, ``'huge'``, ``'highres'``.
    min_mass : float, optional
        Minimum halo mass in Msun/h. Halos below this threshold are
        discarded after loading. Default: no cut.
    cleaned : bool, optional
        Use the cleaned CompaSO halo catalogues (recommended). Default: True.
    fields : list of str, optional
        Additional halo fields to include in the returned dict. Must be
        valid ``CompaSOHaloCatalog`` field names (see abacusutils docs).
        The fields required by jax-hod (``x_L2com``, ``v_L2com``,
        ``r100_L2com``, ``N``) are always loaded.

    Returns
    -------
    dict with keys:

        ``positions``  : ndarray, shape (N, 3)
            Halo centre-of-mass positions in Mpc/h (periodic box coordinates).
        ``masses``     : ndarray, shape (N,)
            Halo masses in Msun/h, computed as N_particles × particle_mass.
        ``radii``      : ndarray, shape (N,)
            Halo virial radii (r100 of the L2 subhalo) in Mpc/h.
        ``velocities`` : ndarray, shape (N, 3)
            Halo centre-of-mass velocities in km/s.
        ``header``     : dict
            Simulation metadata from the ASDF file header, including
            ``BoxSize`` (Mpc/h), ``ParticleMassHMsun``, ``Redshift``, etc.

    Raises
    ------
    ImportError
        If ``abacusutils`` is not installed.
    FileNotFoundError
        If the expected halo catalogue path does not exist.

    Examples
    --------
    >>> from jaxhod.simulations import load_abacus_halos
    >>> halos = load_abacus_halos(
    ...     sim_dir='/path/to/AbacusSummit',
    ...     cosmology='c000',
    ...     phase='ph000',
    ...     redshift=0.5,
    ...     min_mass=1e12,
    ... )
    >>> halos['positions'].shape   # (N_halos, 3)
    >>> halos['masses'].shape      # (N_halos,)
    """
    try:
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    except ImportError:
        raise ImportError(
            'abacusutils is required to load AbacusSummit halos. '
            'Install it with: pip install abacusutils'
        )

    sim_dir = Path(sim_dir)
    sim_prefix = _SIM_TYPES.get(sim_type)
    if sim_prefix is None:
        raise ValueError(
            f"Unknown sim_type '{sim_type}'. "
            f"Valid options: {list(_SIM_TYPES)}"
        )
    sim_name = f'{sim_prefix}_{cosmology}_{phase}'
    halo_path = sim_dir / sim_name / 'halos' / f'z{redshift:.3f}'

    if not halo_path.exists():
        raise FileNotFoundError(
            f'Halo catalogue not found at:\n  {halo_path}\n'
            'Check sim_dir, cosmology, phase, redshift and sim_type.'
        )

    # Always load the fields needed by jax-hod; append any extras.
    required_fields = ['x_L2com', 'v_L2com', 'r100_L2com', 'N']
    load_fields = required_fields + [f for f in (fields or []) if f not in required_fields]

    cat = CompaSOHaloCatalog(
        halo_path,
        cleaned=cleaned,
        fields=load_fields,
        convert_units=True,   # positions in Mpc/h, velocities in km/s
    )

    halos = cat.halos
    particle_mass = cat.header['ParticleMassHMsun']   # Msun/h

    positions  = np.array(halos['x_L2com'],   dtype=np.float32)   # (N, 3) Mpc/h
    velocities = np.array(halos['v_L2com'],   dtype=np.float32)   # (N, 3) km/s
    radii      = np.array(halos['r100_L2com'],dtype=np.float32)   # (N,)   Mpc/h
    masses     = np.array(halos['N'],         dtype=np.float32) * particle_mass  # Msun/h

    # Apply optional mass cut.
    if min_mass is not None:
        mask = masses >= min_mass
        positions  = positions[mask]
        velocities = velocities[mask]
        radii      = radii[mask]
        masses     = masses[mask]

    result = {
        'positions':  positions,
        'masses':     masses,
        'radii':      radii,
        'velocities': velocities,
        'header':     dict(cat.header),
    }

    # Include any extra fields the user requested.
    for field in (fields or []):
        if field not in result:
            data = np.array(halos[field])
            if min_mass is not None:
                data = data[mask]
            result[field] = data

    return result
