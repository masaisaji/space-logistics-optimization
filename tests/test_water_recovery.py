import pytest
from src.input_data_class import MissionParameters

base = dict(
    n_mis=2,
    n_sc_design=1,
    n_sc_per_design=6,
    t_mis_tot=13,
    t_surf_mis=3,
    n_crew=4,
    sample_mass=1000,
    habit_pl_mass=2000,
)


def test_default_is_open_loop():
    # default inputs keep the published consumption model unchanged
    mis = MissionParameters(**base)
    assert mis.net_consumption_cost == pytest.approx(8.655)


def test_water_recovery_reduces_only_the_water_term():
    mis = MissionParameters(**base, water_recovery_rate=0.9)
    # only 90% of the 6.37 kg water term is removed; the rest is still shipped
    assert mis.net_consumption_cost == pytest.approx(8.655 - 0.9 * 6.37)


def test_full_recovery_leaves_food_and_oxygen():
    mis = MissionParameters(**base, water_recovery_rate=1.0)
    assert mis.net_consumption_cost == pytest.approx(8.655 - 6.37)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"water_recovery_rate": -0.1},
        {"water_recovery_rate": 1.1},
        {"water_consumption": -1.0},
        {"water_consumption": 9.0},
    ],
)
def test_rejects_invalid_water_inputs(kwargs):
    with pytest.raises(AssertionError):
        MissionParameters(**base, **kwargs)
