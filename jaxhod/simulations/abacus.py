"""
Readers for AbacusSummit halo catalogues.

Two entry points are provided:

``load_abacus_halos``
    Reads the full CompaSO halo catalogue from ASDF files.  Simple and
    accurate, but requires loading the entire snapshot (tens of GB for
    AbacusSummit base).

``load_abacus_subsampled_halos``
    Reads the pre-generated HOD subsample produced by
    ``abacusnbody.hod.prepare_sim``.  The subsample is a small fraction of
    the full catalogue (a few percent by halo count) stored as slab-wise HDF5
    files, making this the memory-efficient path for HOD work.

Reference: Maksimova et al. 2021 (https://arxiv.org/abs/2110.11398)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


# Simulation name templates for the AbacusSummit suite.
# sim_type -> string prefix as used in the directory names.
_SIM_TYPES = {
    'base':    'AbacusSummit_base',    # 2000 Mpc/h, 6912^3 particles
    'small':   'AbacusSummit_small',   # 500  Mpc/h, 1728^3 particles
    'large':   'AbacusSummit_large',   # 2000 Mpc/h, 10240^3 particles (few sims)
    'huge':    'AbacusSummit_huge',    # 7500 Mpc/h, 8640^3 particles
    'highres': 'AbacusSummit_highres', # 1000 Mpc/h, 6912^3 particles
}


def load_abacus_halos(
    sim_dir: str | Path,
    cosmology: str,
    phase: str,
    redshift: float,
    sim_type: str = 'base',
    min_mass: float | None = None,
    cleaned: bool = True,
    fields: list[str] | None = None,
) -> dict[str, Any]:
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

        ``positions``  : np.ndarray, shape (N, 3), float32
            Halo centre-of-mass positions in Mpc/h (periodic box coordinates).
        ``masses``     : np.ndarray, shape (N,), float32
            Halo masses in Msun/h, computed as N_particles × particle_mass.
        ``radii``      : np.ndarray, shape (N,), float32
            Halo virial radii (r100 of the L2 subhalo) in Mpc/h.
        ``velocities`` : np.ndarray, shape (N, 3), float32
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
    ValueError
        If ``sim_type`` is not one of the recognised AbacusSummit variants.

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

    positions  = np.array(halos['x_L2com'],    dtype=np.float32)  # (N, 3) Mpc/h
    velocities = np.array(halos['v_L2com'],    dtype=np.float32)  # (N, 3) km/s
    radii      = np.array(halos['r100_L2com'], dtype=np.float32)  # (N,)   Mpc/h
    masses     = np.array(halos['N'],          dtype=np.float32) * particle_mass  # Msun/h

    # Apply optional mass cut.
    if min_mass is not None:
        mask = masses >= min_mass
        positions  = positions[mask]
        velocities = velocities[mask]
        radii      = radii[mask]
        masses     = masses[mask]

    result: dict[str, Any] = {
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


def load_abacus_subsampled_halos(
    subsample_dir: str | Path,
    sim_dir: str | Path,
    cosmology: str,
    phase: str,
    redshift: float,
    sim_type: str = 'base',
    mt: bool = False,
    seed: int = 600,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load halo positions, masses, radii and velocities from an AbacusHOD
    subsample, ready to be passed directly to ``populate()``.

    The subsample is produced by ``abacusnbody.hod.prepare_sim`` (from the
    ``abacusutils`` package) before running AbacusHOD.  It is a
    mass-dependent probabilistic downsample of the full halo catalogue stored
    as slab-wise HDF5 files, and is typically 50–100× smaller than the full
    CompaSO catalogue — making it the recommended input for HOD work on large
    simulations.

    Parameters
    ----------
    subsample_dir : str or Path
        Root directory that was passed as ``subsample_dir`` to
        ``prepare_sim``.  The actual slab files live under
        ``{subsample_dir}/{sim_name}/z{redshift:.3f}/``.
    sim_dir : str or Path
        Path to the original AbacusSummit simulation directory (the one
        containing ``{sim_name}/halos/``).  Used *only* to read the ASDF
        file header for ``ParticleMassHMsun`` and ``BoxSize``; no halo
        data is loaded from this path.
    cosmology : str
        Cosmology tag, e.g. ``'c000'`` (Planck 2018 LCDM) or ``'c001'``.
    phase : str
        Phase tag, e.g. ``'ph000'``.  For secondary cosmologies that have
        only a single phase, use ``'ph000'``.
    redshift : float
        Snapshot redshift, e.g. ``0.5``.  Must match the ``z_mock`` used
        when running ``prepare_sim``.
    sim_type : str, optional
        Simulation volume/resolution type.  One of ``'base'`` (default),
        ``'small'``, ``'large'``, ``'huge'``, ``'highres'``.
    mt : bool, optional
        If ``True``, load the multi-tracer (MT) subsample files, which are
        generated when ELG or QSO tracers are enabled in ``prepare_sim``.
        If ``False`` (default), load the LRG-only files.
    seed : int, optional
        Random seed embedded in the subsample filename.  Matches the
        ``--newseed`` argument passed to ``prepare_sim`` (default 600).
    fields : list of str, optional
        Extra halo fields from the HDF5 dataset to include in the returned
        dict.  Available fields beyond the defaults are: ``'r25_L2com'``,
        ``'r90_L2com'``, ``'sigmav3d_L2com'``, ``'deltac_rank'``,
        ``'fenv_rank'``, ``'shear_rank'``, ``'randoms'``.

    Returns
    -------
    dict with keys:

        ``positions``  : np.ndarray, shape (N, 3), float32
            Halo centre-of-mass positions in Mpc/h.
        ``masses``     : np.ndarray, shape (N,), float32
            Halo masses in Msun/h (= particle count × particle mass).
        ``radii``      : np.ndarray, shape (N,), float32
            Halo virial radii using r98 of the L2 subhalo, in Mpc/h.
            This matches what AbacusHOD uses internally.
        ``velocities`` : np.ndarray, shape (N, 3), float32
            Halo centre-of-mass velocities in km/s.
        ``weights``    : np.ndarray, shape (N,), float32
            Inverse subsampling probability (``multi_halos`` field).
            High-mass halos have weight ≈ 1; low-mass halos may have
            weight > 1 because only a fraction of them were kept in the
            subsample.  Pass these to ``populate()`` as ``halo_weights``
            for unbiased number-density estimates.
        ``header``     : dict
            Simulation metadata including ``BoxSize`` (Mpc/h),
            ``ParticleMassHMsun``, ``Redshift``, etc.

    Raises
    ------
    ImportError
        If ``h5py`` or ``asdf`` are not installed (both ship with
        ``abacusutils``).
    FileNotFoundError
        If the subsample directory or ASDF header files are not found.
    ValueError
        If ``sim_type`` is not one of the recognised AbacusSummit variants.

    Notes
    -----
    The subsample files are named::

        halos_xcom_{i}_seed{seed}_abacushod_oldfenv[_MT]_new.h5

    where ``i`` is the slab index (0, 1, 2, …).  Multiple slabs are read
    and concatenated automatically.

    Because low-mass halos are probabilistically downsampled, the subsample
    is not a complete catalogue below the HOD threshold.  When passing the
    result to ``populate()``, no additional ``min_mass`` cut is needed —
    the subsampling scheme in ``prepare_sim`` already removes halos that
    cannot host galaxies for typical LRG/ELG parameters.

    Examples
    --------
    >>> from jaxhod.simulations import load_abacus_subsampled_halos
    >>> halos = load_abacus_subsampled_halos(
    ...     subsample_dir='/path/to/subsamples',
    ...     sim_dir='/path/to/AbacusSummit',
    ...     cosmology='c000',
    ...     phase='ph000',
    ...     redshift=0.5,
    ... )
    >>> halos['positions'].shape   # (N_subsample, 3) — much smaller than full cat
    >>> halos['weights'][:5]       # inverse subsampling probabilities
    """
    try:
        import asdf
        import h5py
    except ImportError:
        raise ImportError(
            'h5py and asdf are required to load AbacusHOD subsamples. '
            'Install them with: pip install abacusutils'
        )

    subsample_dir = Path(subsample_dir)
    sim_dir       = Path(sim_dir)
    z_str         = f'z{redshift:.3f}'

    sim_prefix = _SIM_TYPES.get(sim_type)
    if sim_prefix is None:
        raise ValueError(
            f"Unknown sim_type '{sim_type}'. "
            f"Valid options: {list(_SIM_TYPES)}"
        )
    sim_name = f'{sim_prefix}_{cosmology}_{phase}'

    slab_dir = subsample_dir / sim_name / z_str

    if not slab_dir.exists():
        raise FileNotFoundError(
            f'Subsample directory not found:\n  {slab_dir}\n'
            'Run abacusnbody.hod.prepare_sim first, or check subsample_dir, '
            'cosmology, phase, sim_type, and redshift.'
        )

    # ------------------------------------------------------------------
    # Read the particle mass from the simulation ASDF header.
    # We open just the first slab lazily — no halo data is loaded.
    # ------------------------------------------------------------------
    halo_info_dir = sim_dir / sim_name / 'halos' / z_str / 'halo_info'
    halo_info_files = sorted(halo_info_dir.glob('*.asdf'))
    if not halo_info_files:
        raise FileNotFoundError(
            f'No ASDF halo-info files found in:\n  {halo_info_dir}\n'
            'Check sim_dir and sim_name.'
        )
    with asdf.open(str(halo_info_files[0]), lazy_load=True) as af:
        header = dict(af['header'])
    particle_mass = header['ParticleMassHMsun']   # Msun/h

    # ------------------------------------------------------------------
    # Discover and sort slab files by their integer index.
    # ------------------------------------------------------------------
    mt_tag   = '_MT' if mt else ''
    glob_pat = f'halos_xcom_*_seed{seed}_abacushod_oldfenv{mt_tag}_new.h5'
    slab_files = sorted(
        slab_dir.glob(glob_pat),
        key=lambda p: int(re.search(r'halos_xcom_(\d+)_', p.name).group(1)),
    )
    if not slab_files:
        raise FileNotFoundError(
            f'No subsample halo files matching\n  {glob_pat}\nfound in\n  {slab_dir}\n'
            f'Check mt={mt} and seed={seed}.'
        )

    # ------------------------------------------------------------------
    # Load and concatenate all slabs.
    # ------------------------------------------------------------------
    _default_fields = {'x_L2com', 'v_L2com', 'N', 'r98_L2com', 'multi_halos'}
    extra_fields = [f for f in (fields or []) if f not in _default_fields]

    positions_list:  list[np.ndarray] = []
    velocities_list: list[np.ndarray] = []
    masses_list:     list[np.ndarray] = []
    radii_list:      list[np.ndarray] = []
    weights_list:    list[np.ndarray] = []
    extras:          dict[str, list[np.ndarray]] = {f: [] for f in extra_fields}

    for slab_path in slab_files:
        with h5py.File(slab_path, 'r') as f:
            h = f['halos']
            positions_list.append(np.array(h['x_L2com'],    dtype=np.float32))
            velocities_list.append(np.array(h['v_L2com'],   dtype=np.float32))
            masses_list.append(
                np.array(h['N'], dtype=np.float32) * particle_mass
            )
            radii_list.append(np.array(h['r98_L2com'],    dtype=np.float32))
            weights_list.append(np.array(h['multi_halos'], dtype=np.float32))
            for field in extra_fields:
                extras[field].append(np.array(h[field]))

    result: dict[str, Any] = {
        'positions':  np.concatenate(positions_list,  axis=0),
        'masses':     np.concatenate(masses_list,     axis=0),
        'radii':      np.concatenate(radii_list,      axis=0),
        'velocities': np.concatenate(velocities_list, axis=0),
        'weights':    np.concatenate(weights_list,    axis=0),
        'header':     header,
    }
    for field in extra_fields:
        result[field] = np.concatenate(extras[field], axis=0)

    return result
