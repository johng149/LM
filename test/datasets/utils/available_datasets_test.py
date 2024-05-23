from src.datasets.utils.available_datasets import available_datasets


def test_available_datasets_num():
    assert len(available_datasets) == 2
