import config


def test_repeats_and_temperature_defaults():
    assert config.REPEATS_PER_CASE == 1
    assert config.TARGET_TEMPERATURE == 0.0
