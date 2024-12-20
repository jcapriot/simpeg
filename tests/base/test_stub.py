import pytest
import pathlib


def test_SimPEG_import():
    with pytest.warns(
        FutureWarning,
        match="Importing `SimPEG` is deprecated. please import from `simpeg`.",
    ):
        from SimPEG import data
    import SimPEG
    import simpeg

    assert SimPEG is simpeg
    data_filepath = pathlib.Path(data.__file__)
    assert data_filepath.parent.name == "simpeg"
    assert data_filepath.name == "data.py"
