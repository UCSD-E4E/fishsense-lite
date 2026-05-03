"""Activity to resolve the per-image inputs stage 9 needs for a dive.

Returns a fully-populated `PreprocessSlateImagesInput` ready to hand
to the data-worker's child workflow. Image set is filtered to species
labels with `content_of_image == SLATE_CONTENT_MARKER` whose image
has no DiveSlateLabel row at all — once populate seeds a (possibly
incomplete) row, the slate JPEG is on the file-exchange and we don't
regenerate it. Matches the API selector predicate.

Slate metadata (id, dpi, reference_points) travels alongside the
image set so the data-worker renders the PDF-composite overlay
without an extra fishsense-api call.
"""

from __future__ import annotations

from fishsense_shared import PreprocessSlateImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

SLATE_CONTENT_MARKER = "Slate, Laser on slate"


@activity.defn
async def resolve_slate_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessSlateImagesInput:
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
            raise ValueError(
                f"camera_id={dive.camera_id} has no intrinsics"
            )

        all_slates = await fs.dive_slates.get() or []
        slate = next(
            (s for s in all_slates if s.id == dive.dive_slate_id), None
        )
        if slate is None:
            raise ValueError(
                f"dive_slate_id={dive.dive_slate_id} not found"
            )
        if slate.dpi is None or not slate.reference_points:
            raise ValueError(
                f"dive_slate_id={slate.id} missing dpi or reference_points"
            )

        species_labels = await fs.labels.get_species_labels(dive_id) or []
        existing_slate_labels = (
            await fs.labels.get_dive_slate_labels(dive_id) or []
        )
        labeled_ids = {label.image_id for label in existing_slate_labels}

        target_image_ids = [
            label.image_id
            for label in species_labels
            if label.content_of_image == SLATE_CONTENT_MARKER
            and label.image_id not in labeled_ids
        ]

        images = await fs.images.get(dive_id=dive_id) or []
        checksum_by_id = {image.id: image.checksum for image in images}
        image_checksums = [
            checksum_by_id[image_id]
            for image_id in target_image_ids
            if image_id in checksum_by_id
        ]

        return PreprocessSlateImagesInput(
            dive_id=dive_id,
            image_checksums=image_checksums,
            slate_id=slate.id,
            slate_dpi=slate.dpi,
            reference_points=list(slate.reference_points),
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )
