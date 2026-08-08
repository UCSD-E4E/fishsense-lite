"""Activity to resolve the per-image inputs the slate-detector stage needs.

Returns a fully-populated `PredictSlateImagesInput` for the data-worker's
`PredictSlateImagesWorkflow`. Image set mirrors the API selector
(`select_next_for_slate_prediction`): a slate frame
(`SpeciesLabel.content_of_image='Slate, Laser on slate'`) needs a prediction
only if it has no `SlatePrediction` and no *completed, non-superseded*
`DiveSlateLabel` — so re-runs return only still-unpredicted frames and the model
never re-predicts a human-labeled one. Slate template metadata travels alongside
so the data-worker renders the template without an extra fishsense-api call.
"""

from __future__ import annotations

from typing import List

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.slate_prediction import SlatePrediction
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_shared import PredictSlateImage, PredictSlateImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

SLATE_CONTENT_MARKER = "Slate, Laser on slate"


def _select_frames_needing_prediction(
    species_labels: List[SpeciesLabel],
    predictions: List[SlatePrediction],
    slate_labels: List[DiveSlateLabel],
) -> List[int]:
    """Slate-frame image ids with no prediction and no live human label.

    "Live human label" = `completed AND NOT superseded`, matching the API
    selector. Keys on `completed`, NOT project_id: populate seeds placeholder
    rows that carry a project_id, and a project-id check would starve the
    detector on populate-seeded-but-unlabeled frames.
    """
    predicted_ids = {p.image_id for p in predictions}
    labeled_ids = {
        label.image_id
        for label in slate_labels
        if label.completed and not label.superseded
    }
    return [
        label.image_id
        for label in species_labels
        if label.content_of_image == SLATE_CONTENT_MARKER
        and label.image_id not in predicted_ids
        and label.image_id not in labeled_ids
    ]


@activity.defn
async def resolve_slate_predict_inputs_activity(
    dive_id: int,
) -> PredictSlateImagesInput:
    activity.logger.info("resolving slate predict inputs dive_id=%d", dive_id)
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")
        if dive.dive_slate_id is None:
            raise ValueError(f"dive_id={dive_id} has no dive_slate_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        all_slates = await fs.dive_slates.get() or []
        slate = next((s for s in all_slates if s.id == dive.dive_slate_id), None)
        if slate is None:
            raise ValueError(f"dive_slate_id={dive.dive_slate_id} not found")
        if slate.dpi is None or not slate.reference_points:
            raise ValueError(
                f"dive_slate_id={slate.id} missing dpi or reference_points"
            )

        species_labels = await fs.labels.get_species_labels(dive_id) or []
        predictions = await fs.labels.get_slate_predictions(dive_id) or []
        slate_labels = await fs.labels.get_dive_slate_labels(dive_id) or []
        needing_ids = _select_frames_needing_prediction(
            species_labels, predictions, slate_labels
        )

        images = await fs.images.get(dive_id=dive_id) or []

        # Canonical frames only. The same physical frames live under several

        # dive rows (half of prod's image table is duplicate content), and

        # `is_canonical` marks which copy is the real one. The cohort selectors

        # gate on it, and CLAUDE.md requires resolvers to mirror the selector

        # predicate exactly -- otherwise the dispatched per-image work would not

        # match what the cohort promised, and the dive could never drain.

        # This also covers on-demand/backfill runs, which bypass the cohort.

        images = [image for image in images if image.is_canonical]
        checksum_by_id = {image.id: image.checksum for image in images}

        activity.logger.info(
            "resolved slate predict inputs dive_id=%d slate_id=%d needing=%d",
            dive_id,
            slate.id,
            len(needing_ids),
        )
        return PredictSlateImagesInput(
            dive_id=dive_id,
            slate_id=slate.id,
            slate_name=slate.name or "",
            dpi=slate.dpi,
            template_points=[list(p) for p in slate.reference_points],
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
            images=[
                PredictSlateImage(image_id=image_id, checksum=checksum_by_id[image_id])
                for image_id in needing_ids
                if image_id in checksum_by_id
            ],
        )
