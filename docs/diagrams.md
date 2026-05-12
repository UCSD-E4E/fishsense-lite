# fishsense-lite-mono — architecture diagrams

UML-flavored Mermaid diagrams. GitHub renders Mermaid natively in
markdown — open this file on GitHub or in any Mermaid-aware editor.

These diagrams are derived from current code; if a diagram and the
code disagree, the code is right and the diagram is stale. Treat them
as a starting orientation, not a contract.

## 1. System context (component diagram)

How the four services, two libs, and external systems wire together.

```mermaid
flowchart LR
    subgraph EXT[External systems]
        LS[Label Studio]
        NAS["E4E NAS<br/>(Synology, FileStation HTTPS)"]
        TC["Temporal cluster"]
    end

    subgraph ORCH["Orchestrator host"]
        API["fishsense-api<br/>(FastAPI)"]
        APIWW["fishsense-api-workflow-worker<br/>queue: fishsense_api_queue"]
        BUW["fishsense-backup-worker<br/>queue: fishsense_backup_queue"]
        PG[("PostgreSQL<br/>fishsense, superset, temporal_db")]
        FX["nginx static_file_server<br/>file-exchange (DAV)"]
        SUP["Superset"]
    end

    subgraph NRP["Kubernetes — NRP/Nautilus now<br/>(Junkyard / Qualcomm: future targets)"]
        DPW["fishsense-data-processing-workflow-worker<br/>queue: fishsense_data_processing_queue<br/>(Deployment, scale-to-zero)"]
    end

    subgraph LIBS["Workspace libraries"]
        SHARED["fishsense-shared<br/>(config, logging, mTLS, exception_group)"]
        SDK["fishsense-api-sdk<br/>(async HTTP client)"]
    end

    API   --->|asyncpg| PG
    APIWW --->|sdk| API
    APIWW --->|label-studio-sdk| LS
    APIWW --->|gRPC mTLS| TC
    APIWW -.->|k8s scale 0..N| DPW
    DPW   --->|sdk| API
    DPW   --->|gRPC mTLS| TC
    DPW   --->|GET raw / PUT jpeg<br/>(authentik basic-auth)| FX
    BUW   --->|pg_dump -Fc| PG
    BUW   --->|FileStation HTTPS| NAS
    BUW   --->|gRPC mTLS| TC
    SUP   --->|read-only| PG

    %% Library deps (workspace)
    API   -.->|imports| SHARED
    APIWW -.->|imports| SHARED
    APIWW -.->|imports| SDK
    DPW   -.->|imports| SHARED
    DPW   -.->|imports| SDK
    BUW   -.->|imports| SHARED
    SDK   -.->|HTTP| API
```

**Why backup-worker is its own service.** Narrower blast radius —
only `pg_dump`-equivalent DB credentials, only NAS write access,
separate task queue, separate image. Mixing it into the
data-processing worker would broaden either side's privileges.

## 2. Deployment topology

What runs where, and which composes file owns what. Maps to the files
in [deploy/](../deploy/).

```mermaid
flowchart TB
    subgraph PROD["Prod orchestrator host"]
        direction TB
        subgraph C_BASE["compose.yml + compose.orchestrator.yml"]
            P_API["fishsense-api"]
            P_FX["static_file_server nginx DAV"]
            P_PG[("PostgreSQL 17")]
            P_AUTHENTIK["Authentik OAuth proxy"]
        end
        subgraph C_TEMPORAL["compose.temporal.yml"]
            P_TEMPORAL["Temporal cluster"]
        end
        subgraph C_WORKERS["compose.workers.yml"]
            P_APIWW["fishsense-api-workflow-worker"]
            P_BUW["fishsense-backup-worker"]
        end
        subgraph C_SUPERSET["compose.superset.yml"]
            P_SUP["Superset"]
        end
    end

    subgraph NRP["NRP / Nautilus (k8s) - deploy/k8s/data-worker/<br/>Junkyard / Qualcomm: future targets"]
        D_DPW["fishsense-data-processing-workflow-worker<br/>Deployment (replicas managed by the api-worker)"]
    end

    subgraph DEV["Local devcontainer - compose.local.yml"]
        L_PG[("postgres 17")]
        L_TEMPORAL["temporal auto-setup"]
        L_API["fishsense-api pinned image"]
        L_FX["nginx static_file_server"]
        L_LS["label-studio (with hard-coded admin token)"]
        L_DEV["dev workspace bind-mount"]
    end

    subgraph CI["GitHub Actions"]
        CI_BUILD["build.yml"]
        CI_RELEASE["release.yml"]
        CI_PROMOTE["promote.yml"]
        CI_DEPLOY["deploy.yml"]
        GHCR["GHCR images"]
        GH_REL["GitHub release tag"]
        AUTO_PR["auto-deploy PR"]
    end

    CI_BUILD   -->|on push to main| GHCR
    CI_RELEASE -->|on release PR merge| GH_REL
    GH_REL     -->|release published event| CI_PROMOTE
    CI_PROMOTE --> GHCR
    CI_PROMOTE -->|opens image-pin PR| AUTO_PR
    AUTO_PR    -->|merge: orchestrator pin| CI_DEPLOY
    AUTO_PR    -->|merge: data-worker pin| CI_DEPLOY
    CI_DEPLOY  -->|docker compose up -d| PROD
    CI_DEPLOY  -->|kubectl apply -k| NRP

    P_APIWW -.->|k8s scale 0..N| D_DPW
```

Notes:

- The deploy PR is intentional — a human reviews the image-pin diff
  (compose `image:` for the orchestrator stack, kustomize `newTag:` for
  the data-worker) before any prod change. See
  [.github/workflows/deploy.yml](../.github/workflows/deploy.yml).
- The orchestrator deploy job uses a persistent ops-managed checkout
  (path passed via repo variable `DEPLOY_DIR`); volumes/secrets sit
  beside the compose files as untracked siblings. The data-worker
  deploy job is a GitHub-hosted `kubectl apply -k` using the
  `NRP_KUBECONFIG` secret — no persistent dir; config/certs live in
  cluster ConfigMaps/Secrets.
- The local stack does **not** layer onto `compose.yml` — Authentik /
  mTLS / letsencrypt aren't bootable on a laptop.

## 3. Domain class diagram (Postgres tables)

Authoritative SQLModel definitions in
[services/fishsense-api/src/fishsense_api/models/](../services/fishsense-api/src/fishsense_api/models/).
The SDK side
([libs/fishsense-api-sdk/src/fishsense_api_sdk/models/](../libs/fishsense-api-sdk/src/fishsense_api_sdk/models/))
hand-mirrors these as Pydantic models; drift is policed by
`tests/test_sdk_drift.py`.

```mermaid
classDiagram
    direction LR

    class Camera {
        +int id PK
        +str serial_number UQ
        +str name UQ
    }
    class CameraIntrinsics {
        +int id PK
        +List camera_matrix
        +List distortion_coefficients
        +int camera_id FK
    }
    class LaserExtrinsics {
        +int id PK
        +List laser_position
        +List laser_axis
        +datetime created_at
        +int dive_id FK
        +int camera_id FK
    }

    class Dive {
        +int id PK
        +str name
        +str path UQ
        +datetime dive_datetime
        +Priority priority
        +bool flip_dive_slate
        +int camera_id FK
        +int dive_slate_id FK
    }
    class DiveSlate {
        +int id PK
        +str name UQ
        +int dpi
        +str path UQ
        +datetime created_at
        +List reference_points
    }

    class Image {
        +int id PK
        +str path UQ
        +datetime taken_datetime
        +str checksum
        +bool is_canonical
        +int dive_id FK
        +int camera_id FK
    }

    class DiveFrameCluster {
        +int id PK
        +DataSource data_source
        +datetime updated_at
        +int dive_id FK
        +int fish_id FK
    }
    class DiveFrameClusterImageMapping {
        +int dive_frame_cluster_id PK_FK
        +int image_id PK_FK
    }

    class Fish {
        +int id PK
        +int species_id FK
    }
    class Species {
        +int id PK
        +str scientific_name
        +str common_name
    }
    class Measurement {
        +int id PK
        +float length_m
        +int image_id FK
        +int fish_id FK
    }

    class User {
        +int id PK
        +int label_studio_id UQ
        +str email UQ
        +str first_name
        +str last_name
        +datetime last_activity
        +datetime date_joined
    }

    class LaserLabel {
        +int id PK
        +int label_studio_task_id UQ
        +int label_studio_project_id
        +float x
        +float y
        +str label
        +bool superseded
        +bool completed
        +Dict label_studio_json
        +int image_id FK
        +int user_id FK
    }
    class HeadTailLabel {
        +int id PK
        +int label_studio_task_id UQ
        +int label_studio_project_id
        +float head_x
        +float head_y
        +float tail_x
        +float tail_y
        +bool superseded
        +bool completed
        +Dict label_studio_json
        +int image_id FK
        +int user_id FK
    }
    class SpeciesLabel {
        +int id PK
        +int label_studio_task_id UQ
        +int label_studio_project_id
        +str grouping
        +str fish_measurable_category
        +str fish_angle_category
        +str fish_curved_category
        +bool completed
        +Dict label_studio_json
        +int image_id FK
        +int user_id FK
    }
    class DiveSlateLabel {
        +int id PK
        +int label_studio_task_id UQ
        +int label_studio_project_id
        +bool upside_down
        +List reference_points
        +List slate_rectangle
        +List skipped_points
        +bool completed
        +Dict label_studio_json
        +int image_id FK
        +int user_id FK
    }

    class DataSource {
        <<enum>>
        PREDICTION
        LABEL_STUDIO
    }
    class Priority {
        <<enum>>
        LOW
        HIGH
    }

    Camera "1" <-- "*" CameraIntrinsics : camera_id
    Camera "1" <-- "*" Dive : camera_id
    Camera "1" <-- "*" Image : camera_id
    Camera "1" <-- "*" LaserExtrinsics : camera_id
    DiveSlate "1" <-- "*" Dive : dive_slate_id
    Dive "1" <-- "*" Image : dive_id
    Dive "1" <-- "*" DiveFrameCluster : dive_id
    Dive "1" <-- "*" LaserExtrinsics : dive_id
    Fish "1" <-- "*" DiveFrameCluster : fish_id
    Species "1" <-- "*" Fish : species_id
    Image "1" <-- "*" Measurement : image_id
    Fish  "1" <-- "*" Measurement : fish_id
    DiveFrameCluster "*" -- "*" Image : DiveFrameClusterImageMapping
    Image "1" <-- "*" LaserLabel : image_id
    Image "1" <-- "*" HeadTailLabel : image_id
    Image "1" <-- "*" SpeciesLabel : image_id
    Image "1" <-- "*" DiveSlateLabel : image_id
    User "1" <-- "*" LaserLabel : user_id
    User "1" <-- "*" HeadTailLabel : user_id
    User "1" <-- "*" SpeciesLabel : user_id
    User "1" <-- "*" DiveSlateLabel : user_id
    DiveFrameCluster ..> DataSource : uses
    Dive ..> Priority : uses
```

Unique-image-per-Label-Studio-project is enforced via composite
`UniqueConstraint(image_id, label_studio_project_id)` on each of the
four label tables.

## 4. SDK class diagram

Façade in [libs/fishsense-api-sdk/src/fishsense_api_sdk/client.py](../libs/fishsense-api-sdk/src/fishsense_api_sdk/client.py).
Each sub-client owns its own `httpx.AsyncClient` (opened in
`__aenter__`, closed in `__aexit__`). All sub-clients share a single
`asyncio.Semaphore` so `max_concurrent_requests` caps concurrency
across resources, not per-resource.

```mermaid
classDiagram
    direction LR

    class Client {
        +str base_url
        -Semaphore __semaphore
        +CameraClient cameras
        +DiveClient dives
        +DiveSlateClient dive_slates
        +FishClient fish
        +ImageClient images
        +LabelClient labels
        +UserClient users
        +__init__(base_url, username, password, timeout, max_concurrent_requests)
        +__aenter__() Client
        +__aexit__(exc_type, exc_value, traceback) None
    }

    class ClientBase {
        <<abstract>>
        -httpx.AsyncClient __client_internal
        -bool __inside_context
        -Semaphore __semaphore
        +Logger logger
        +__init__(base_url, username, password, timeout, semaphore)
        +__aenter__()
        +__aexit__()
    }

    class CameraClient
    class DiveClient
    class DiveSlateClient
    class FishClient
    class ImageClient
    class LabelClient
    class UserClient

    Client *-- CameraClient
    Client *-- DiveClient
    Client *-- DiveSlateClient
    Client *-- FishClient
    Client *-- ImageClient
    Client *-- LabelClient
    Client *-- UserClient

    CameraClient    --|> ClientBase
    DiveClient      --|> ClientBase
    DiveSlateClient --|> ClientBase
    FishClient      --|> ClientBase
    ImageClient     --|> ClientBase
    LabelClient     --|> ClientBase
    UserClient      --|> ClientBase
```

## 5. Sequence — data-worker per-image preprocess

The shape every stage in the data-worker shares (stages 0.1, 2, 5.1,
9). Stage 9 also fetches a slate PDF; otherwise identical.

```mermaid
sequenceDiagram
    autonumber
    participant T as Temporal
    participant W as PreprocessWorkflow
    participant A as preprocess_image_activity
    participant FX as nginx_file_exchange
    participant FC as fishsense_core
    participant CV as opencv

    T->>W: schedule workflow
    W->>A: execute_activity(payload)
    A->>FX: GET /api/v1/exchange/raw/{checksum}.ORF
    FX-->>A: raw bytes
    opt stage 9 only
        A->>FX: GET /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf
        FX-->>A: pdf bytes
    end
    A->>A: asyncio.to_thread(_rectify_overlay_encode)
    A->>FC: RawImage(bytes) → RectifiedImage(intrinsics)
    FC-->>A: rectified ndarray
    A->>CV: stage-specific overlay (text / rectangle / pdf composite)
    A->>CV: cv2.imencode(.jpg)
    CV-->>A: jpeg bytes
    A->>FX: PUT /api/v1/exchange/{folder}/{checksum}.JPG
    FX-->>A: 201/204
    A-->>W: None
    W-->>T: WorkflowResult
```

The `_rectify_overlay_encode` step is sync CPU work, kept out of the
event loop via `asyncio.to_thread`. Pure-logic
`overlay_and_encode_jpeg` is broken out so unit tests don't need
Temporal, httpx, or fishsense-core decode.

## 6. Sequence — api-worker label sync

Hourly `SyncLabelStudioLaserLabelsWorkflow` (the head/tail variant is
isomorphic). Runs entirely on the orchestrator host.

```mermaid
sequenceDiagram
    autonumber
    participant T as Temporal
    participant W as SyncLabelStudioLaserLabelsWorkflow
    participant ALS as get_laser_project_ids_activity
    participant ASYNC as sync_users_activity
    participant API as fishsense_api
    participant LS as label_studio
    participant ASY as sync_laser_labels_activity

    T->>W: schedule fires (every 1h)
    W->>ASYNC: execute_activity (mirror Label Studio users → fishsense-api)
    ASYNC->>LS: list users
    LS-->>ASYNC: users
    ASYNC->>API: upsert User rows
    ASYNC-->>W: ok

    W->>ALS: execute_activity (resolve laser project IDs)
    ALS->>API: list projects
    API-->>ALS: project ids
    ALS-->>W: List[int]

    loop per project_id
        W->>ASY: execute_activity(project_id)
        ASY->>LS: list completed tasks for project_id
        LS-->>ASY: tasks (with annotations)
        ASY->>API: upsert LaserLabel rows<br/>(unique by image_id+project_id)
        ASY-->>W: ok
    end
    W-->>T: WorkflowResult
```

## 7. Sequence — Label Studio project create + populate

The eight on-demand workflows in the api-worker — Create + Populate
× {Laser, Species, HeadTail, DiveSlate}. Same shape per stage; laser
shown.

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant T as Temporal
    participant CW as CreateLaserLabelStudioProjectWorkflow
    participant CA as create_laser_label_studio_project_activity
    participant LS as label_studio
    participant PW as PopulateLaserLabelStudioProjectWorkflow
    participant GA as get_active_laser_label_studio_project_ids_activity
    participant API as fishsense_api
    participant PA as populate_laser_label_studio_project_activity

    Note over U,LS: Create — once per deployment, idempotent
    U->>T: workflow start CreateLaserLabelStudioProjectWorkflow
    T->>CW: run()
    CW->>CA: execute_activity()
    CA->>LS: list projects, find by title
    alt title exists
        LS-->>CA: project_id
    else not found
        CA->>LS: create from <STAGE>_LABELING_CONFIG_XML
        LS-->>CA: project_id
    end
    CA-->>CW: project_id
    CW-->>T: project_id

    Note over U,API: Populate — per dive, on demand
    U->>T: workflow start PopulateLaserLabelStudioProjectWorkflow(dive_id)
    T->>PW: run(dive_id)
    PW->>GA: execute_activity()
    GA->>API: SDK get_laser_label_studio_project_ids(incomplete=True)
    API-->>GA: List[int]
    GA-->>PW: project_ids

    par Semaphore(4) — bounded fan-out
        loop per project_id
            PW->>PA: execute_activity(dive_id, project_id)
            PA->>API: list images for dive
            API-->>PA: images
            PA->>LS: import tasks
            LS-->>PA: ok
            PA-->>PW: count
        end
    end
    PW-->>T: total tasks imported
```

Bootstrap caveat: `incomplete=True` returns nothing for a brand-new
project (zero labels), so Populate is a no-op until something seeds
at least one label. Existing prod projects already have labels;
fresh deployments need a seed step that doesn't exist yet.

## 8. CI/CD pipeline (build → release → promote → deploy)

State of an image as it moves from a commit to prod.

```mermaid
stateDiagram-v2
    [*] --> Pushed
    Pushed --> Built : build.yml builds and tags image
    Built --> ReleasePR : release-please opens PR
    ReleasePR --> Tagged : PR merged so release.yml cuts tag
    Tagged --> Promoted : promote.yml retags image
    Promoted --> AutoDeployPR : promote.yml opens image-pin PR
    AutoDeployPR --> Deployed : human merges so deploy.yml deploys
    Deployed --> [*]

    note right of Promoted
        Manifest-only retag, no layer push.
        Race-proof - tags the image built from the
        release commit, not whatever latest happens
        to be.
    end note

    note right of AutoDeployPR
        Orchestrator-stack services: bumps the
        image: pin in deploy/compose*.yml, deploy.yml
        runs docker compose up -d on fishsense-prod.
        data-processing-workflow-worker: bumps the
        newTag: in deploy/k8s/data-worker/kustomization.yaml,
        deploy.yml runs kubectl apply -k on NRP.
    end note
```

## Editing & rendering

- These are Mermaid fences, rendered by GitHub on `*.md` files. Local
  preview: VS Code's *Markdown Preview Mermaid Support* extension or
  the Mermaid Live Editor (https://mermaid.live).
- Keep diagrams aligned with code on the same PR that changes the
  shape. A diagram that drifts is worse than no diagram — readers will
  trust it.
- New service or table: add a node here in the same PR that adds the
  Dockerfile or the SQLModel.
