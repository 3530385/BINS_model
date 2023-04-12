import pytest
from model_bins import BINS
import numpy as np

g = 9.81
U = np.deg2rad(15) / 3600


def test_init(bins_with_random, bins_without_random):
    assert bins_with_random.psi == np.deg2rad(90)
    assert bins_with_random.random_proc
    assert bins_without_random.deltapsi == np.deg2rad(-0.5)
    assert not bins_without_random.random_proc


def test_fail_init():
    with pytest.raises(TypeError):
        bins = BINS(psi=90, theta=3, gamma=0,
                    dpsi=-0.5, dwbx=1, dwby=0.5,
                    dabx=1, daby=0.5,  # sigma_a=1,
                    Tka=0.2, sigma_w=2, Tkw=0.1,
                    rand=True)


def test_ideal_vistavka():
    psi, theta, gamma = 10, 20, 30

    bins = BINS(psi=psi, theta=theta, gamma=gamma,
                dpsi=-0.5, dwbx=1, dwby=0.5,
                dabx=1, daby=0.5, sigma_a=1,
                Tka=0.2, sigma_w=2, Tkw=0.1,
                rand=True)
    psi, theta, gamma = np.deg2rad(np.array([10, 20, 30]))
    c11 = np.cos(gamma) * np.cos(psi) + np.sin(theta) * np.sin(gamma) * np.sin(psi)
    c12 = np.sin(psi) * np.cos(theta)
    c13 = np.cos(psi) * np.sin(gamma) - np.sin(psi) * np.sin(theta) * np.cos(gamma)
    c21 = -np.sin(psi) * np.cos(gamma) + np.cos(psi) * np.sin(theta) * np.sin(gamma)
    c22 = np.cos(psi) * np.cos(theta)
    c23 = -np.sin(psi) * np.sin(gamma) - np.cos(psi) * np.sin(theta) * np.cos(gamma)
    c31 = -np.cos(theta) * np.sin(gamma)
    c32 = np.sin(theta)
    c33 = np.cos(theta) * np.cos(gamma)
    for c_i, c_j in zip([c11, c12, c13,
                         c21, c22, c23,
                         c31, c32, c33], bins.ideal_vistavka().flatten()):
        assert c_i == c_j


def test_get_measurement(bins_with_random, bins_without_random):
    _, mean = bins_with_random.get_measurement("accel")
    c_ob = bins_with_random.ideal_vistavka().T
    measurement_without_noise = c_ob @ np.array([0, 0, g]) + bins_with_random.dab
    assert np.allclose(mean, measurement_without_noise, atol=.001)

    _, mean = bins_without_random.get_measurement("accel")
    measurement_without_noise = c_ob @ np.array([0, 0, g]) + bins_without_random.dab
    assert np.allclose(mean, measurement_without_noise)

    _, mean = bins_without_random.get_measurement("gyro")
    measurement_without_noise = c_ob @ np.array(
        [0, U * np.cos(np.deg2rad(56)), U * np.sin(np.deg2rad(56))]) + bins_without_random.dwb
    assert np.allclose(mean, measurement_without_noise)

    _, mean = bins_with_random.get_measurement("gyro")
    measurement_without_noise = c_ob @ np.array(
        [0, U * np.cos(np.deg2rad(56)), U * np.sin(np.deg2rad(56))]) + bins_without_random.dwb
    assert np.allclose(mean, measurement_without_noise, atol=.001)

    with pytest.raises(RuntimeError):
        _, mean = bins_with_random.get_measurement("gyroscope")


def test_real_vistavka(bins_with_random, bins_without_random):
    _, mean_a = bins_without_random.get_measurement("accel")
    c_bo_align = bins_without_random.real_vistavka()
    theta_zk = np.arctan(c_bo_align[2, 1] / np.sqrt(c_bo_align[0, 1] ** 2 + c_bo_align[1, 1] ** 2))
    gamma_zk = -np.arctan(c_bo_align[2, 0] / c_bo_align[2, 2])
    real_errors = [theta_zk - bins_without_random.theta, gamma_zk - bins_without_random.gamma]
    theory_errors = [bins_without_random.dab[1] / np.sqrt(g ** 2 - mean_a[1] ** 2),
                     (bins_without_random.dab[2] * mean_a[0] - bins_without_random.dab[0] * mean_a[2]) / (
                                 mean_a[0] ** 2 + mean_a[2] ** 2)]
    assert np.allclose(real_errors, theory_errors)

    _, mean_a = bins_with_random.get_measurement("accel")
    c_bo_align = bins_with_random.real_vistavka()
    theta_zk = np.arctan(c_bo_align[2, 1] / np.sqrt(c_bo_align[0, 1] ** 2 + c_bo_align[1, 1] ** 2))
    gamma_zk = -np.arctan(c_bo_align[2, 0] / c_bo_align[2, 2])
    real_errors = [theta_zk - bins_with_random.theta, gamma_zk - bins_with_random.gamma]
    theory_errors = [bins_with_random.dab[1] / np.sqrt(g ** 2 - mean_a[1] ** 2),
                     (bins_with_random.dab[2] * mean_a[0] - bins_with_random.dab[0] * mean_a[2]) / (
                             mean_a[0] ** 2 + mean_a[2] ** 2)]
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"{real_errors}")
    print(f"{theory_errors}")
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    assert np.allclose(real_errors, theory_errors, atol=.001)



