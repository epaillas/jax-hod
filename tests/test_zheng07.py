import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxhod import Zheng07, populate, NFW, UniformSphere
from jaxhod.populate import _populate


@pytest.fixture
def model():
    return Zheng07(
        log_Mmin=13.0,
        sigma_logM=0.5,
        log_M0=13.0,
        log_M1=14.0,
        alpha=1.0,
    )


@pytest.fixture
def halos():
    key = jax.random.PRNGKey(0)
    n = 500
    masses = 10 ** jax.random.uniform(key, (n,), minval=12.0, maxval=15.5)
    positions = jax.random.uniform(key, (n, 3), minval=0.0, maxval=1000.0)
    radii = (masses / 1e14) ** (1.0 / 3.0)
    return positions, masses, radii


class TestZheng07:
    def test_mean_ncen_range(self, model):
        masses = jnp.logspace(11, 16, 100)
        ncen = model.mean_ncen(masses)
        assert jnp.all(ncen >= 0.0)
        assert jnp.all(ncen <= 1.0)

    def test_mean_ncen_monotonic(self, model):
        masses = jnp.logspace(11, 16, 100)
        ncen = model.mean_ncen(masses)
        # Allow float32 rounding tolerance at the saturating tails
        assert jnp.all(jnp.diff(ncen) >= -1e-6)

    def test_mean_ncen_midpoint(self, model):
        # At log10(M) = log_Mmin, mean occupation should be exactly 0.5
        M_mid = 10.0 ** model.log_Mmin
        ncen = model.mean_ncen(jnp.array([M_mid]))
        assert jnp.isclose(ncen[0], 0.5)

    def test_mean_nsat_nonnegative(self, model):
        masses = jnp.logspace(11, 16, 100)
        nsat = model.mean_nsat(masses)
        assert jnp.all(nsat >= 0.0)

    def test_mean_nsat_zero_below_cutoff(self, model):
        low_masses = jnp.logspace(10, model.log_M0 - 0.5, 20)
        nsat = model.mean_nsat(low_masses)
        assert jnp.all(nsat == 0.0)

    def test_mean_nsat_increasing(self, model):
        masses = jnp.logspace(model.log_M0 + 0.1, 16, 50)
        nsat = model.mean_nsat(masses)
        assert jnp.all(jnp.diff(nsat) >= 0.0)


class TestPopulate:
    """Tests for the public populate() API."""

    def test_output_keys(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(1))
        assert set(result.keys()) == {'positions', 'is_central'}

    def test_output_shapes_consistent(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(2))
        n_gal = result['positions'].shape[0]
        assert result['positions'].shape == (n_gal, 3)
        assert result['is_central'].shape == (n_gal,)

    def test_nonzero_galaxies(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(3))
        assert result['positions'].shape[0] > 0

    def test_centrals_at_halo_positions(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(4))
        cen_pos = result['positions'][result['is_central']]
        halo_pos = np.array(positions)
        # Each central must sit exactly on a halo centre
        for cp in cen_pos:
            assert np.any(np.all(np.isclose(halo_pos, cp), axis=1))

    def test_default_profile_is_nfw(self, model, halos):
        positions, masses, radii = halos
        key = jax.random.PRNGKey(5)
        result_default = populate(positions, masses, radii, model, key)
        result_nfw = populate(positions, masses, radii, model, key, profile=NFW())
        np.testing.assert_array_equal(result_default['positions'], result_nfw['positions'])

    def test_uniform_sphere_profile(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(6),
                          profile=UniformSphere())
        assert result['positions'].shape[0] > 0

    def test_nfw_profiles_differ_from_uniform(self, model, halos):
        positions, masses, radii = halos
        key = jax.random.PRNGKey(7)
        r_nfw = populate(positions, masses, radii, model, key, profile=NFW())
        r_uni = populate(positions, masses, radii, model, key, profile=UniformSphere())
        assert not np.allclose(r_nfw['positions'], r_uni['positions'])

    def test_nfw_per_halo_concentration(self, model, halos):
        positions, masses, radii = halos
        n = masses.shape[0]
        concentrations = jnp.full((n,), 8.0)
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(8),
                          profile=NFW(concentration=concentrations))
        assert result['positions'].shape[0] > 0


class TestPopulateInternal:
    """Tests for _populate() that verify the padded internal structure."""

    def test_output_shapes(self, model, halos):
        positions, masses, radii = halos
        n, max_sat = masses.shape[0], 10
        result = _populate(positions, masses, radii, model, jax.random.PRNGKey(10),
                           max_satellites=max_sat)
        total = n + n * max_sat
        assert result['positions'].shape == (total, 3)
        assert result['is_central'].shape == (total,)
        assert result['mask'].shape == (total,)

    def test_satellites_within_virial_radius(self, model, halos):
        positions, masses, radii = halos
        n, max_sat = masses.shape[0], 20
        result = _populate(positions, masses, radii, model, jax.random.PRNGKey(11),
                           max_satellites=max_sat)
        sat_positions = result['positions'][n:].reshape(n, max_sat, 3)
        sat_mask = result['mask'][n:].reshape(n, max_sat)
        offsets = sat_positions - np.array(positions)[:, None, :]
        distances = jnp.linalg.norm(offsets, axis=-1)
        valid_distances = distances[sat_mask]
        expected_radii = jnp.repeat(radii[:, None], max_sat, axis=1)[sat_mask]
        assert jnp.all(valid_distances <= expected_radii + 1e-5)

    def test_no_satellites_without_central(self, model, halos):
        positions, masses, radii = halos
        n, max_sat = masses.shape[0], 10
        result = _populate(positions, masses, radii, model, jax.random.PRNGKey(12),
                           max_satellites=max_sat)
        is_central = result['is_central'][:n]
        sat_mask = result['mask'][n:].reshape(n, max_sat)
        has_satellites = sat_mask.any(axis=1)
        assert jnp.all(~has_satellites | is_central)

    def test_nfw_satellites_within_virial_radius(self, model, halos):
        positions, masses, radii = halos
        n, max_sat = masses.shape[0], 20
        result = _populate(positions, masses, radii, model, jax.random.PRNGKey(13),
                           max_satellites=max_sat, profile=NFW(concentration=5.0))
        sat_positions = result['positions'][n:].reshape(n, max_sat, 3)
        sat_mask = result['mask'][n:].reshape(n, max_sat)
        offsets = sat_positions - np.array(positions)[:, None, :]
        distances = jnp.linalg.norm(offsets, axis=-1)
        valid_distances = distances[sat_mask]
        expected_radii = jnp.repeat(radii[:, None], max_sat, axis=1)[sat_mask]
        assert jnp.all(valid_distances <= expected_radii + 1e-5)

    def test_jit_compatible(self, model, halos):
        positions, masses, radii = halos
        jitted = jax.jit(
            lambda pos, m, r, k: _populate(pos, m, r, model, k, max_satellites=10)
        )
        result = jitted(positions, masses, radii, jax.random.PRNGKey(14))
        assert result['mask'].sum() > 0

    def test_nfw_jit_compatible(self, model, halos):
        positions, masses, radii = halos
        profile = NFW(concentration=5.0)
        jitted = jax.jit(
            lambda pos, m, r, k: _populate(pos, m, r, model, k,
                                           max_satellites=10, profile=profile)
        )
        result = jitted(positions, masses, radii, jax.random.PRNGKey(15))
        assert result['mask'].sum() > 0
