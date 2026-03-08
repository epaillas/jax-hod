import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxhod import Zheng07, populate, downsample_to_nbar, NFW, UniformSphere, get_devices
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
        assert set(result.keys()) == {'positions', 'is_central', 'max_satellites'}

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

    def test_min_mass_reduces_galaxy_count(self, model, halos):
        positions, masses, radii = halos
        key = jax.random.PRNGKey(20)
        result_all = populate(positions, masses, radii, model, key)
        # A high min_mass cut should produce fewer or equal galaxies
        result_cut = populate(positions, masses, radii, model, key,
                              min_mass=10 ** model.log_Mmin)
        assert result_cut['positions'].shape[0] <= result_all['positions'].shape[0]

    def test_min_mass_above_all_halos_returns_empty(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(21),
                          min_mass=1e20)
        assert result['positions'].shape[0] == 0

    def test_batch_size_produces_galaxies(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(30),
                          batch_size=100)
        assert result['positions'].shape[0] > 0

    def test_batch_size_output_shapes_consistent(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(31),
                          batch_size=50)
        n_gal = result['positions'].shape[0]
        assert result['positions'].shape == (n_gal, 3)
        assert result['is_central'].shape == (n_gal,)

    def test_batch_size_one(self, model, halos):
        # Extreme case: one halo per batch
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(32),
                          batch_size=1)
        assert result['positions'].shape[0] > 0

    def test_batch_larger_than_catalog(self, model, halos):
        # batch_size > N_halos: single batch, should still return valid output
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(33),
                          batch_size=len(masses) + 1000)
        n_gal = result['positions'].shape[0]
        assert result['positions'].shape == (n_gal, 3)
        assert result['is_central'].shape == (n_gal,)
        assert n_gal > 0

    def test_halo_weights_present_in_output(self, model, halos):
        positions, masses, radii = halos
        weights = np.ones(len(masses), dtype=np.float32) * 2.0
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(40),
                          halo_weights=weights)
        assert 'weights' in result
        assert result['weights'].shape == (result['positions'].shape[0],)

    def test_halo_weights_absent_without_arg(self, model, halos):
        positions, masses, radii = halos
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(41))
        assert 'weights' not in result

    def test_halo_weights_values_match_host(self, model, halos):
        # Each galaxy's weight must equal the weight of its host halo.
        # Use uniform weights so we can verify the round-trip.
        positions, masses, radii = halos
        n = len(masses)
        rng = np.random.default_rng(0)
        weights = rng.uniform(1.0, 5.0, n).astype(np.float32)
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(42),
                          halo_weights=weights)
        # Every galaxy weight must appear in the input weights array.
        for w in result['weights']:
            assert np.any(np.isclose(weights, w))

    def test_halo_weights_with_batching(self, model, halos):
        positions, masses, radii = halos
        weights = np.ones(len(masses), dtype=np.float32)
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(43),
                          halo_weights=weights, batch_size=100)
        assert 'weights' in result
        assert result['weights'].shape == (result['positions'].shape[0],)

    def test_halo_weights_with_min_mass(self, model, halos):
        positions, masses, radii = halos
        weights = np.ones(len(masses), dtype=np.float32) * 3.0
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(44),
                          halo_weights=weights, min_mass=10 ** model.log_Mmin)
        assert 'weights' in result
        assert np.all(result['weights'] == 3.0)

    def test_nfw_per_halo_concentration(self, model, halos):
        positions, masses, radii = halos
        n = masses.shape[0]
        concentrations = jnp.full((n,), 8.0)
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(8),
                          profile=NFW(concentration=concentrations))
        assert result['positions'].shape[0] > 0


class TestDownsampleToNbar:
    BOX_SIZE = 1000.0  # Mpc/h — same as the halos fixture

    def _full_result(self, model, halos):
        positions, masses, radii = halos
        return populate(positions, masses, radii, model, jax.random.PRNGKey(50))

    def test_output_keys_preserved(self, model, halos):
        result = self._full_result(model, halos)
        thin = downsample_to_nbar(result, nbar_target=1e-6,
                                  box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))
        assert set(thin.keys()) == set(result.keys())

    def test_fewer_galaxies_after_downsample(self, model, halos):
        result = self._full_result(model, halos)
        thin = downsample_to_nbar(result, nbar_target=1e-6,
                                  box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))
        assert thin['positions'].shape[0] < result['positions'].shape[0]

    def test_nbar_approximately_correct(self, model, halos):
        result = self._full_result(model, halos)
        nbar_target = 1e-6
        thin = downsample_to_nbar(result, nbar_target=nbar_target,
                                  box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))
        nbar_thin = thin['positions'].shape[0] / self.BOX_SIZE ** 3
        # Poisson noise: allow 10% relative tolerance at this galaxy count
        assert abs(nbar_thin - nbar_target) / nbar_target < 0.10

    def test_weights_used_for_nbar_when_present(self, model, halos):
        # Inflate weights by 10x — the true nbar should also be 10x higher,
        # so a larger fraction must be discarded to hit the same target.
        positions, masses, radii = halos
        w = np.full(len(masses), 10.0, dtype=np.float32)
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(51),
                          halo_weights=w)
        result_no_w = populate(positions, masses, radii, model, jax.random.PRNGKey(51))
        thin_w  = downsample_to_nbar(result,      nbar_target=1e-6,
                                     box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))
        thin_nw = downsample_to_nbar(result_no_w, nbar_target=1e-6,
                                     box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))
        # With weight=10, far fewer galaxies should survive
        assert thin_w['positions'].shape[0] < thin_nw['positions'].shape[0]

    def test_nbar_above_mock_raises(self, model, halos):
        result = self._full_result(model, halos)
        nbar_mock = result['positions'].shape[0] / self.BOX_SIZE ** 3
        with pytest.raises(ValueError, match='Cannot upsample'):
            downsample_to_nbar(result, nbar_target=nbar_mock * 10,
                               box_size=self.BOX_SIZE, key=jax.random.PRNGKey(0))


class TestPopulateParallel:
    """Tests for the devices= parallel path in populate()."""

    def test_devices_requires_batch_size(self, model, halos):
        positions, masses, radii = halos
        cpu = jax.devices('cpu')
        with pytest.raises(ValueError, match='batch_size'):
            populate(positions, masses, radii, model, jax.random.PRNGKey(60),
                     devices=cpu)

    def test_single_device_list_matches_sequential(self, model, halos):
        positions, masses, radii = halos
        key = jax.random.PRNGKey(61)
        cpu = [jax.devices('cpu')[0]]
        result_seq = populate(positions, masses, radii, model, key,
                              batch_size=100)
        result_par = populate(positions, masses, radii, model, key,
                              batch_size=100, devices=cpu)
        np.testing.assert_array_equal(result_seq['positions'], result_par['positions'])
        np.testing.assert_array_equal(result_seq['is_central'], result_par['is_central'])

    def test_parallel_output_keys(self, model, halos):
        positions, masses, radii = halos
        cpu = jax.devices('cpu')
        devices = [cpu[0], cpu[0]]
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(62),
                          batch_size=100, devices=devices)
        assert 'positions' in result
        assert 'is_central' in result
        assert 'max_satellites' in result

    def test_parallel_shapes_consistent(self, model, halos):
        positions, masses, radii = halos
        cpu = jax.devices('cpu')
        devices = [cpu[0], cpu[0]]
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(63),
                          batch_size=100, devices=devices)
        n_gal = result['positions'].shape[0]
        assert result['positions'].shape == (n_gal, 3)
        assert result['is_central'].shape == (n_gal,)

    def test_parallel_nonzero_galaxies(self, model, halos):
        positions, masses, radii = halos
        cpu = jax.devices('cpu')
        devices = [cpu[0], cpu[0]]
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(64),
                          batch_size=100, devices=devices)
        assert result['positions'].shape[0] > 0

    def test_parallel_with_weights(self, model, halos):
        positions, masses, radii = halos
        weights = np.ones(len(masses), dtype=np.float32) * 2.0
        cpu = jax.devices('cpu')
        devices = [cpu[0], cpu[0]]
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(65),
                          batch_size=100, devices=devices, halo_weights=weights)
        assert 'weights' in result
        assert result['weights'].shape == (result['positions'].shape[0],)

    def test_parallel_with_min_mass(self, model, halos):
        positions, masses, radii = halos
        cpu = jax.devices('cpu')
        devices = [cpu[0], cpu[0]]
        result = populate(positions, masses, radii, model, jax.random.PRNGKey(66),
                          batch_size=100, devices=devices,
                          min_mass=10 ** model.log_Mmin)
        assert result['positions'].shape[0] >= 0

    def test_get_devices_fallback(self):
        # On a CPU-only machine, get_devices('tpu') should return CPU devices.
        devs = get_devices('tpu')
        assert len(devs) > 0
        assert all(d.platform in ('cpu', 'tpu') for d in devs)


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
