

class TestModelVersionTagIsAnIdempotencyKey:
    """`backfill_headtail_predictions` keys on `(task_id, model_version)`, so
    the tag has to be a pure function of the stage's behaviour.

    It used to interpolate the checkpoint's pod-local filesystem path, so
    moving the cache directory produced a different key for byte-identical
    output and stacked a duplicate prediction onto every seeded task.
    """

    def test_is_stable_across_calls(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        assert headtail_model_version_tag() == headtail_model_version_tag()

    def test_carries_no_filesystem_path(self):
        from fishsense_shared.headtail_predictor import headtail_model_version_tag

        tag = headtail_model_version_tag()
        assert "/" not in tag
        assert "checkpoint=" not in tag

    def test_names_the_behaviour_version_and_crop(self):
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_CROP_HEIGHT,
            HEADTAIL_CROP_WIDTH,
            HEADTAIL_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        tag = headtail_model_version_tag()
        assert f"v{HEADTAIL_PREDICTOR_VERSION}" in tag
        assert f"{HEADTAIL_CROP_WIDTH}x{HEADTAIL_CROP_HEIGHT}" in tag
