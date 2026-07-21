"""Shared helpers for populate / create LS-project activities.

Each populate stage creates LS tasks pointing at the labeler-facing
static-file server, imports them in one batch, then upserts a
per-image label row anchoring the (image, LS task, LS project) triple.
Each create stage uses `create_or_get_label_studio_project` to
idempotently materialize a per-dive LS project from a stored
labeling-config XML; populate workflows call create first to get the
target project ID and then push tasks into it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable, Iterable, List

from label_studio_sdk.client import LabelStudio
from label_studio_sdk.core import ApiError
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    get_fs_client,
    get_ls_client,
)
from fishsense_api_workflow_worker.config import settings


# Title given to the per-project Garage S3 source storage. Matching on
# (bucket, title) makes registration idempotent across re-runs.
LS_S3_STORAGE_TITLE = "garage"


def build_image_url(folder: str, checksum: str) -> str:
    """Build the `s3://` URI for a preprocessed JPEG in Garage.

    Label Studio resolves this to a presigned GET URL at serve time via
    the per-project S3 source storage (see
    `ensure_label_studio_s3_storage`). `folder` is the **physical**
    Garage prefix the data-worker wrote to (preprocess_jpeg /
    preprocess_groups_jpeg / preprocess_headtail_jpeg /
    preprocess_slate_images_jpeg) — i.e. the exact key the JPEG lives
    at. There's no virtual->physical rewrite layer anymore, which is
    why the old nginx alias mismatch (issue #113) is gone by
    construction.

    JPEGs live in the **labels** bucket (LS-facing) under the optional
    `labels_prefix`, separate from the raw/slate scratch `bucket`. Falls
    back to `bucket`/no-prefix for single-bucket layouts.
    """
    obj = settings.object_store
    bucket = obj.get("labels_bucket", None) or obj.bucket
    prefix = (obj.get("labels_prefix", "") or "").strip("/")
    key = f"{prefix}/{folder}/{checksum}.JPG" if prefix else f"{folder}/{checksum}.JPG"
    return f"s3://{bucket}/{key}"


def _ls_s3_presign_credentials() -> tuple[str, str]:
    """`(access_key, secret_key)` Label Studio uses to presign GET URLs.

    Prefers the optional read-only `presign_*` key; falls back to the
    main object-store key when ops haven't scoped a separate one.
    """
    obj = settings.object_store
    access = obj.get("presign_access_key", None) or obj.access_key
    secret = obj.get("presign_secret_key", None) or obj.secret_key
    return access, secret


async def ensure_label_studio_s3_storage(project_id: int) -> None:
    """Idempotently register a Garage S3 *source* storage on
    `project_id` so LS presigns the `s3://` image URIs in each task's
    data.

    Open-source LS storages are per-project and projects are per-dive,
    so every freshly-created project needs this. Matches on
    (bucket, title) to avoid duplicate registrations on re-runs. Does
    NOT call `.sync()` — tasks are POSTed explicitly; this connection
    exists only to enable presigning.
    """
    ls = _get_ls_client()
    obj = settings.object_store
    # LS serves JPEGs from the labels bucket; register storage against it
    # (not the scratch bucket). Prefix scopes it within the shared labels
    # bucket, mirroring the coral-gardeners layout.
    bucket = obj.get("labels_bucket", None) or obj.bucket
    prefix = (obj.get("labels_prefix", "") or "").strip("/")

    existing = await asyncio.to_thread(
        lambda: list(ls.import_storage.s3.list(project=project_id))
    )
    for storage in existing:
        if (
            getattr(storage, "bucket", None) == bucket
            and getattr(storage, "title", None) == LS_S3_STORAGE_TITLE
        ):
            return

    access_key, secret_key = _ls_s3_presign_credentials()
    create_kwargs = {
        "project": project_id,
        "title": LS_S3_STORAGE_TITLE,
        "bucket": bucket,
        "s3endpoint": settings.object_store.endpoint_url,
        "region_name": settings.object_store.region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "presign": True,
        "use_blob_urls": False,
    }
    if prefix:
        create_kwargs["prefix"] = prefix
    await asyncio.to_thread(lambda: ls.import_storage.s3.create(**create_kwargs))
    activity.logger.info(
        "registered LS S3 storage project_id=%d bucket=%s", project_id, bucket
    )


def _canonical_label_config(config: str | None) -> str | None:
    """Structure-only form of a labeling-config XML, or None if unparseable.

    LS reformats `label_config` server-side (indentation, self-closing
    style, attribute order), so a raw string compare reports drift on every
    single run and would re-PATCH the project hourly forever. Comparing
    parsed structure instead means we only write when the *choices* really
    changed.
    """
    if not config or not isinstance(config, str):
        return None
    try:
        root = ET.fromstring(config)
    except ET.ParseError:
        return None

    def _node(element):
        return (
            element.tag,
            tuple(sorted((k, (v or "").strip()) for k, v in element.attrib.items())),
            tuple(_node(child) for child in element),
        )

    return repr(_node(root))


def _label_config_differs(current: str | None, desired: str) -> bool:
    """True when `current` needs to be replaced by `desired`."""
    canonical_current = _canonical_label_config(current)
    canonical_desired = _canonical_label_config(desired)
    if canonical_current is None or canonical_desired is None:
        # Unparseable on either side — fall back to a whitespace-normalized
        # compare rather than guessing (and rather than looping on a PATCH
        # that can never converge).
        raw_current = current if isinstance(current, str) else ""
        return " ".join(raw_current.split()) != " ".join(desired.split())
    return canonical_current != canonical_desired


async def heal_labeling_config(ls: LabelStudio, project: Any, desired_xml: str) -> bool:
    """Push `desired_xml` onto an already-created project when it drifted.

    Without this, editing a `<STAGE>_LABELING_CONFIG_XML` constant only
    affects projects created *after* the deploy — every existing per-dive
    project keeps the config it was born with, so a taxonomy change (e.g.
    swapping the Fish Model choices) silently never reaches annotators.

    Returns True when the config was rewritten.
    """
    current = getattr(project, "label_config", None)
    if not isinstance(current, str):
        # `projects.list` may omit the config; fetch the detail view.
        detail = await asyncio.to_thread(lambda: ls.projects.get(id=project.id))
        current = getattr(detail, "label_config", None)

    if not _label_config_differs(current, desired_xml):
        return False

    try:
        await asyncio.to_thread(
            lambda: ls.projects.update(id=project.id, label_config=desired_xml)
        )
    except ApiError as e:
        # LS rejects a config that would invalidate existing annotations
        # (e.g. dropping a choice value someone already used). Keep the old
        # config and keep populating rather than failing the whole stage.
        activity.logger.warning(
            "Could not update labeling config for LS project id=%d: %s. "
            "Project keeps its previous config; reconcile by hand if the "
            "taxonomy change is required.",
            project.id,
            e,
        )
        return False

    activity.logger.info(
        "Updated labeling config for LS project id=%d (config drift healed)",
        project.id,
    )
    return True


async def create_or_get_label_studio_project(
    *,
    project_title: str,
    labeling_config_xml: str,
) -> int:
    """Idempotent create — return the LS project ID for `project_title`,
    creating one with `labeling_config_xml` if none exists, and
    idempotently registering the Garage S3 source storage on it.

    Used by the create-side activities. Per-dive titles are built by
    `build_per_dive_title`.
    """
    ls = _get_ls_client()
    workspace_id = await asyncio.to_thread(_resolve_workspace_id, ls)

    def _list_titled():
        # Scope the idempotency lookup to the target workspace so a same-
        # titled project in another workspace can't shadow it. `workspaces`
        # (plural) is a server-side filter; None means "all workspaces".
        projects = (
            ls.projects.list(workspaces=[workspace_id])
            if workspace_id is not None
            else ls.projects.list()
        )
        return [p for p in projects if p.title == project_title]

    matches = await asyncio.to_thread(_list_titled)
    if matches:
        if len(matches) > 1:
            activity.logger.warning(
                "Multiple LS projects titled %r; using id=%d",
                project_title,
                matches[0].id,
            )
        project_id = matches[0].id
        # Self-heal: converge an already-created project onto the current
        # labeling-config constant. Editing the constant is otherwise a
        # no-op for every project that already exists.
        if labeling_config_xml:
            await heal_labeling_config(ls, matches[0], labeling_config_xml)
        await ensure_label_studio_s3_storage(project_id)
        return project_id

    if not labeling_config_xml:
        raise RuntimeError(
            f"Cannot create LS project {project_title!r}: the labeling-"
            "config XML constant is empty. Paste the labeling-config XML "
            "from your existing prod project (Project Settings -> Labeling "
            "Interface -> Code) into the corresponding constant."
        )

    # Created as a draft (is_published left at the LS default). Per-dive
    # projects are only published once their task set is complete — see
    # `publish_label_studio_project`, called by the populate activities —
    # so a still-filling or JPEG-deferred project stays hidden from
    # annotators until every intended task exists.
    project = await asyncio.to_thread(
        ls.projects.create,
        title=project_title,
        label_config=labeling_config_xml,
        workspace=workspace_id,
    )
    activity.logger.info(
        "Created LS project %r (id=%d)", project_title, project.id
    )
    await ensure_label_studio_s3_storage(project.id)
    return project.id


# LS rejects `Project.title` over 50 characters with a 400.
LS_PROJECT_TITLE_MAX = 50


async def build_per_dive_title(dive_id: int, suffix: str) -> str:
    """Build a per-dive LS project title `"{dive.name} #{dive_id} - {suffix}"`.

    Used by the four create-LS-project activities so each dive gets its own
    project. `#{dive_id}` is **always** included so dives that share a `name`
    still get distinct projects — dive names are NOT unique in prod (mislabeled
    captures, duplicate-named dives, and same-site/same-camera repeats all
    exist), so keying the title on the name alone silently merged two dives
    into one project.

    `dive.name` is truncated (never the `#{dive_id}` tail) to fit LS's 50-char
    cap, so the id-based uniqueness survives even for long names. A nameless
    dive yields `"#{dive_id} - {suffix}"`.
    """
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
    if dive is None:
        raise RuntimeError(
            f"Cannot build LS project title for dive_id={dive_id}: "
            "no such dive found via the API"
        )
    tail = f"#{dive_id} - {suffix}"
    name = (dive.name or "").strip()
    if not name:
        return tail[:LS_PROJECT_TITLE_MAX]
    budget = LS_PROJECT_TITLE_MAX - len(tail) - 1  # 1 for the space before tail
    if budget < 1:
        return tail[:LS_PROJECT_TITLE_MAX]
    return f"{name[:budget].rstrip()} {tail}"


# Hosted LS's import is asynchronous — the created tasks are usually listable
# immediately, but the import call returns before they're guaranteed queryable.
# Bounded poll so a small lag doesn't force a Temporal activity retry.
_IMPORT_VISIBILITY_ATTEMPTS = 12
_IMPORT_VISIBILITY_INTERVAL_S = 2.0


def _task_image_url(task: dict) -> str | None:
    data = task.get("data", {}) or {}
    return data.get("image") or data.get("img")


def _normalize_image_url(url: str | None) -> str | None:
    """Reduce an LS task image URL to a stable comparison key (the s3 URI).

    Freshly-built task data carries `s3://…`, but when hosted LS *lists* tasks
    it returns `data.image` as a per-task presign resolve-wrapper
    `/tasks/{task_id}/resolve/?fileuri=<base64(s3://…)>`. The embedded task_id
    makes that string unique per task, so comparing it raw against the built
    `s3://` URL never matched — silently defeating dedup and re-importing the
    whole batch every run (projects ballooned to thousands of tasks, 0 rows).
    Decode the `fileuri` back to the s3 URI so both forms compare equal.
    """
    if not url or url.startswith("s3://"):
        return url
    if "fileuri=" in url:
        query = urllib.parse.urlparse(url).query
        fileuri = urllib.parse.parse_qs(query).get("fileuri", [None])[0]
        if fileuri:
            try:
                return base64.b64decode(fileuri).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                return url
    return url


async def import_tasks_and_record_labels(
    *,
    project_id: int,
    tasks: List[dict],
    record_label: Callable[[Any, int], Awaitable[None]],
    items: Iterable[Any],
) -> int:
    """Import `tasks` to LS, then PUT one label row per (item, task_id).

    `record_label(item, task_id)` is the per-stage hook that upserts a
    LaserLabel/SpeciesLabel/HeadTailLabel/DiveSlateLabel row anchoring the
    (image, LS task, project) triple. Rows are PUT in parallel via TaskGroup;
    the api PUTs are upserts so partial-failure replay is safe.

    **Hosted LS (Enterprise/heartex) imports asynchronously**: `import_tasks`
    returns an import-job id, NOT task ids — only OSS returns task ids. The old
    code read `imported.task_ids`, which doesn't exist on hosted LS: it crashed
    *after* the tasks were created but *before* writing any label rows, so every
    Temporal retry re-imported the whole batch (projects ballooned to tens of
    thousands of tasks with zero label rows).

    So we resolve task ids by listing the project's tasks and matching each
    input task by its image URL (checksum-based, unique per image), and we
    **dedupe against tasks already in the project before importing** — a retry
    after a mid-activity failure then re-imports nothing and just writes the
    missing rows.
    """
    tasks = list(tasks)
    items_list = list(items)
    if not tasks:
        return 0
    if len(tasks) != len(items_list):
        raise RuntimeError(
            f"import_tasks_and_record_labels: {len(tasks)} tasks for "
            f"{len(items_list)} items — the two lists must be parallel"
        )

    urls = [_normalize_image_url(_task_image_url(t)) for t in tasks]
    if any(u is None for u in urls):
        raise RuntimeError(
            "import task missing an image URL in `data` — cannot anchor label"
        )

    ls = _get_ls_client()

    def _url_to_task_id() -> dict:
        mapping: dict = {}
        for task in ls.tasks.list(project=project_id):
            data = getattr(task, "data", {}) or {}
            url = _normalize_image_url(data.get("image") or data.get("img"))
            if url is not None:
                mapping[url] = task.id
        return mapping

    # Dedupe against what's already imported so a retry doesn't duplicate.
    known = await asyncio.to_thread(_url_to_task_id)
    to_import = [t for t, u in zip(tasks, urls) if u not in known]

    if to_import:
        await asyncio.to_thread(
            lambda: ls.projects.import_tasks(
                project_id, request=to_import, return_task_ids=True
            )
        )
        want = {u for u in urls if u not in known}
        for _ in range(_IMPORT_VISIBILITY_ATTEMPTS):
            known = await asyncio.to_thread(_url_to_task_id)
            if want <= known.keys():
                break
            activity.heartbeat()
            await asyncio.sleep(_IMPORT_VISIBILITY_INTERVAL_S)
        missing = want - known.keys()
        if missing:
            raise RuntimeError(
                f"{len(missing)} imported task(s) not visible in project "
                f"{project_id} after import — retrying (dedupe prevents dupes)"
            )

    async with asyncio.TaskGroup() as tg:
        for item, url in zip(items_list, urls):
            tg.create_task(record_label(item, known[url]))
            activity.heartbeat()

    return len(items_list)


async def publish_label_studio_project(project_id: int) -> None:
    """Idempotently publish an LS project so annotators can see it.

    Called by the populate activities **only once a project's task set is
    complete** — never at create time. On LS Enterprise a project is created
    as a draft (invisible to annotators); publishing is deferred until every
    intended task exists so a still-filling project (e.g. species images
    whose stage-2 JPEGs haven't been processed yet, which the JPEG gate
    defers) is never shown half-populated. `projects.update` with
    `is_published=True` is idempotent, so re-running populate on an
    already-complete project is a harmless no-op.
    """
    ls = _get_ls_client()
    await asyncio.to_thread(
        lambda: ls.projects.update(id=project_id, is_published=True)
    )
    activity.logger.info("Published LS project id=%d", project_id)


def _get_ls_client() -> LabelStudio:
    """Local indirection over `activities.utils.get_ls_client` so unit
    tests can monkeypatch a single symbol on `populate_utils` and have
    both `create_or_get_label_studio_project` and
    `import_tasks_and_record_labels` pick up the fake client."""
    return get_ls_client()


def _resolve_workspace_id(ls: LabelStudio) -> int | None:
    """Resolve the configured `label_studio.workspace` name to its LS id.

    LS Enterprise groups projects into workspaces; new per-dive projects
    should be created in the tenant's workspace (`FishSense` in prod) rather
    than the caller's personal one. Returns ``None`` when unset (OSS LS /
    local dev / tests) so project creation falls back to the default
    workspace. Raises if a name is configured but no such workspace exists —
    a silent fallback would scatter projects into the wrong workspace.
    """
    name = (settings.label_studio.get("workspace") or "").strip()
    if not name:
        return None
    matches = [w for w in ls.workspaces.list() if w.title == name]
    if not matches:
        raise RuntimeError(
            f"Label Studio workspace {name!r} not found "
            "(settings.label_studio.workspace) — create it or fix the config."
        )
    return matches[0].id
