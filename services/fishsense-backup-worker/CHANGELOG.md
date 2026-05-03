# Changelog

## [0.2.3](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-backup-worker-v0.2.2...fishsense-backup-worker-v0.2.3) (2026-05-03)


### Bug Fixes

* **api-workflow-worker,backup-worker:** force fresh login per NasClient ([5655bbb](https://github.com/UCSD-E4E/fishsense-lite/commit/5655bbb8c4c8e62fe639e09d45362d9e42ea50e7))
* force fresh DSM login per NasClient (synology-api class-cache bug) ([458e966](https://github.com/UCSD-E4E/fishsense-lite/commit/458e966138f6cb5bbce6e53d580c7d6da4c963a7))

## [0.2.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-backup-worker-v0.2.1...fishsense-backup-worker-v0.2.2) (2026-05-01)


### Bug Fixes

* **backup-worker:** create NAS dest dir before upload + surface FileStation errors ([3b8f717](https://github.com/UCSD-E4E/fishsense-lite/commit/3b8f717ea6ad3769c4c4a4adb3d0a714fca56b61))
* **backup-worker:** silence pylint W0613 in pg_dump stderr-capture test ([51a5097](https://github.com/UCSD-E4E/fishsense-lite/commit/51a5097bb24f59eda82449983d87f5aa42ff4aca))
* update config so that postgress can now connect ([abb47c3](https://github.com/UCSD-E4E/fishsense-lite/commit/abb47c36facec910423a16caf86db381e5785b36))

## [0.2.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-backup-worker-v0.2.0...fishsense-backup-worker-v0.2.1) (2026-04-30)


### Bug Fixes

* **backup-worker:** surface pg_dump stderr and drop bogus workflow-id placeholder ([ce4f748](https://github.com/UCSD-E4E/fishsense-lite/commit/ce4f7484ffc1e093afc5ba319c6bc5eca45a2d1c))

## [0.2.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-backup-worker-v0.1.0...fishsense-backup-worker-v0.2.0) (2026-04-29)


### Features

* **backup-worker:** new service for nightly Postgres -&gt; NAS backups ([dc7fa0d](https://github.com/UCSD-E4E/fishsense-lite/commit/dc7fa0d10e97c2f43f663a1402ba280ecca10976))
* **ci:** build-once / promote-tag deploy split + monorepo-aware Dockerfiles ([66d20b9](https://github.com/UCSD-E4E/fishsense-lite/commit/66d20b9b9bc834bc2ebc88461006d71a7dd14faa))


### Bug Fixes

* **lint:** pylint clean across the new worker code ([dbd832f](https://github.com/UCSD-E4E/fishsense-lite/commit/dbd832f61e062677f6477df9c38cf151162aee31))
