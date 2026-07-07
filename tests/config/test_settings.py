from forecasting.config import Config, load_config


def test_config_output_dir_uses_run_id(tmp_path):
    cfg = Config(output_root=str(tmp_path), run_id="run-001")
    assert cfg.output_dir.endswith("run-001")


def test_load_config_reads_toml(tmp_path):
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('data_dir = "dataset"\nrun_id = "abc"\n', encoding="utf-8")

    cfg = load_config(str(cfg_file))

    assert cfg.data_dir == "dataset"
    assert cfg.run_id == "abc"
