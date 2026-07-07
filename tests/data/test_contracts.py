from forecasting.config import Config
from forecasting.data import validate_input_contract


def test_validate_input_contract_detects_missing(tmp_path):
    cfg = Config(data_dir=str(tmp_path), run_id="test")

    try:
        validate_input_contract(cfg)
    except FileNotFoundError as exc:
        assert "master_data.parquet" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_validate_input_contract_returns_paths(tmp_path):
    ind = tmp_path / "master_data.parquet"
    tgt = tmp_path / "target_price.parquet"
    ind.write_bytes(b"data")
    tgt.write_bytes(b"data")

    cfg = Config(data_dir=str(tmp_path), run_id="test")
    ind_path, tgt_path = validate_input_contract(cfg)

    assert ind_path.endswith("master_data.parquet")
    assert tgt_path.endswith("target_price.parquet")
