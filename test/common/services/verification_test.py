from src.common.services.verification import verify_args

def test_verify_args():
    infos = {}
    kwargs = {}
    result, has_error = verify_args(infos, **kwargs)
    assert len(result) == 0