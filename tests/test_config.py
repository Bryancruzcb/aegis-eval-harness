import aegis_eval.core.config as config


def test_repeats_and_temperature_defaults():
    assert config.REPEATS_PER_CASE == 1
    assert config.TARGET_TEMPERATURE == 0.0


def test_base_dir_is_the_repo_root():
    assert (config.BASE_DIR / "aegis_eval").is_dir()
    assert (config.BASE_DIR / "run.py").is_file()
    assert config.CASES_PATH == config.BASE_DIR / "data" / "test_cases.json"
    assert config.CASES_PATH.is_file()
