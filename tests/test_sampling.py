import pytest

from app.volumes.sampling import sample_function

CUBE = [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]


def test_sample_has_requested_resolution():
    grid = sample_function("x**2 + y**2 + z**2 - 1", CUBE, 32)

    assert grid.shape == (32, 32, 32)
    assert grid.origin == (-1.5, -1.5, -1.5)


def test_sample_center_of_sphere_is_inside():
    grid = sample_function("x**2 + y**2 + z**2 - 1", CUBE, 32)

    assert grid.values.min() < -0.99
    assert grid.values[15, 15, 15] < 0


def test_sample_corners_are_outside():
    grid = sample_function("x**2 + y**2 + z**2 - 1", CUBE, 16)

    assert grid.values[0, 0, 0] > 0
    assert grid.values[-1, -1, -1] > 0



def test_first_index_runs_along_x():
    grid = sample_function("x + 10*y + 100*z", [[0, 1], [0, 1], [0, 1]], 5)

    assert grid.values[1, 0, 0] == pytest.approx(0.25)
    assert grid.values[0, 1, 0] == pytest.approx(2.5)
    assert grid.values[0, 0, 1] == pytest.approx(25.0)



def test_spacing_is_distance_between_samples():
    grid = sample_function("x", [[0, 1], [0, 2], [0, 4]], 5)

    assert grid.spacing == (0.25, 0.5, 1.0)