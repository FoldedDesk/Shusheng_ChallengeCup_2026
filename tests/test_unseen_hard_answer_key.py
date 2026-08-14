from scripts.verify_unseen_hard_answers import verify


def test_isolated_hard_holdout_answer_key_is_deterministically_verified():
    report = verify()
    assert report == {
        "rows": 14,
        "verified": 14,
        "passed": True,
        "failed_indices": [],
        "unchecked_indices": [],
        "runtime_answer_exposure": False,
    }
