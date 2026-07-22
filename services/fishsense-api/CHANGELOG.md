# CHANGELOG

<!-- version list -->

## [1.31.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.31.0...fishsense-api-v1.31.1) (2026-07-22)


### Bug Fixes

* **api:** species preprocess selector/view require the image be in a cluster ([#390](https://github.com/UCSD-E4E/fishsense-lite/issues/390)) ([eb4f6f2](https://github.com/UCSD-E4E/fishsense-lite/commit/eb4f6f23bc88be91fc971db387f54637c1bf39fe))

## [1.31.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.30.2...fishsense-api-v1.31.0) (2026-07-22)


### Features

* **label-studio:** hourly labeling-config reconcile for every per-dive project ([#376](https://github.com/UCSD-E4E/fishsense-lite/issues/376)) ([7e8e609](https://github.com/UCSD-E4E/fishsense-lite/commit/7e8e609bfc4c9255e902aea9e5eb25c55df7f04b))


### Bug Fixes

* **api:** stage-14 cohort must skip species rows with no scientific name ([#375](https://github.com/UCSD-E4E/fishsense-lite/issues/375)) ([64f9117](https://github.com/UCSD-E4E/fishsense-lite/commit/64f911747f2e35ce8fdc6389f925da56c646ac30))

## [1.30.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.30.1...fishsense-api-v1.30.2) (2026-07-21)


### Bug Fixes

* **api:** make preprocess cohort gates superseded-aware ([#353](https://github.com/UCSD-E4E/fishsense-lite/issues/353)) ([b3415a5](https://github.com/UCSD-E4E/fishsense-lite/commit/b3415a5df28c697934abf6424548507e414f1b28))

## [1.30.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.30.0...fishsense-api-v1.30.1) (2026-07-21)


### Bug Fixes

* **api:** upsert put_*_label on natural key, not blind INSERT ([#347](https://github.com/UCSD-E4E/fishsense-lite/issues/347)) ([d5d1aa8](https://github.com/UCSD-E4E/fishsense-lite/commit/d5d1aa87aa3da9a29c472807274cfac31c12a1d3))

## [1.30.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.29.1...fishsense-api-v1.30.0) (2026-07-18)


### Features

* **populate-species:** scheduled populate parent (decoupled, superseded-aware) ([#309](https://github.com/UCSD-E4E/fishsense-lite/issues/309)) ([082b27d](https://github.com/UCSD-E4E/fishsense-lite/commit/082b27de33ab7ccc90081d0daefa92b8596afa20))

## [1.29.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.29.0...fishsense-api-v1.29.1) (2026-07-17)


### Bug Fixes

* **stage14:** make fish measurement idempotent and unstick `measured` ([#287](https://github.com/UCSD-E4E/fishsense-lite/issues/287)) ([d57fdec](https://github.com/UCSD-E4E/fishsense-lite/commit/d57fdec7782febcfc9e42894efa7d86d340c420e))

## [1.29.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.28.1...fishsense-api-v1.29.0) (2026-07-14)


### Features

* **labels:** give SpeciesLabel + DiveSlateLabel a `superseded` dead-letter flag ([#247](https://github.com/UCSD-E4E/fishsense-lite/issues/247)) ([f851e93](https://github.com/UCSD-E4E/fishsense-lite/commit/f851e933d4e3b2c8f596b945d43b9fd0124241b6))

## [1.28.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.28.0...fishsense-api-v1.28.1) (2026-07-14)


### Bug Fixes

* **api:** exclude superseded from laser/headtail label-studio-project-ids ([#246](https://github.com/UCSD-E4E/fishsense-lite/issues/246)) ([1354abc](https://github.com/UCSD-E4E/fishsense-lite/commit/1354abcbca27e34b87a4a65f44367d3670edeefd))

## [1.28.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.27.3...fishsense-api-v1.28.0) (2026-05-06)


### Features

* species cascade flip + stage 1 clustering automation ([ec565f5](https://github.com/UCSD-E4E/fishsense-lite/commit/ec565f5ffe4c53c24196c69ff3a2554098034db9))

## [1.27.3](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.27.2...fishsense-api-v1.27.3) (2026-05-05)


### Bug Fixes

* **api:** make 3770d7474078 migration idempotent vs create_all ([0fd59c6](https://github.com/UCSD-E4E/fishsense-lite/commit/0fd59c6010df816858ca73c31a0fb5902658b821))
* **api:** stamp head on fresh DB instead of running historical migrations ([7a60dbd](https://github.com/UCSD-E4E/fishsense-lite/commit/7a60dbdbb571d1cb3585f45ed3592e1c5587e0da))


### Documentation

* TDD convention + service map + READMEs refresh post-mafl ([e844bfe](https://github.com/UCSD-E4E/fishsense-lite/commit/e844bfe823298a772e87d1fa76d0e182dd103fb8))

## [1.27.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.27.1...fishsense-api-v1.27.2) (2026-05-05)


### Bug Fixes

* **api:** make labelstudiosynccursor migration idempotent vs create_all ([dbce3eb](https://github.com/UCSD-E4E/fishsense-lite/commit/dbce3ebce847531316e2c36258aee00ca13b02a8))
* **api:** unblock auto-migrate on prod (alembic runtime dep + idempotent migration) ([aed1f65](https://github.com/UCSD-E4E/fishsense-lite/commit/aed1f65148b42f99f922bd0d9d3d7291c378f42d))

## [1.27.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.27.0...fishsense-api-v1.27.1) (2026-05-04)


### Bug Fixes

* **api:** move alembic to runtime deps (HOTFIX for v1.27.0 startup crash) ([76bb8e6](https://github.com/UCSD-E4E/fishsense-lite/commit/76bb8e65b96c6189db7a685be53710c8b0ab8e2a))
* **api:** move alembic to runtime deps so the auto-migrate import works ([f0c81cd](https://github.com/UCSD-E4E/fishsense-lite/commit/f0c81cdc1ad61f2d3e458502a9b0543d39fb68e5))

## [1.27.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.26.0...fishsense-api-v1.27.0) (2026-05-04)


### Features

* **api:** auto-apply alembic migrations on FastAPI startup ([36ca0fc](https://github.com/UCSD-E4E/fishsense-lite/commit/36ca0fc4bb8702aba6dd43827120f83b07ab7c80))
* **api:** dive_pipeline_status view for Superset dashboards ([5ec1ae9](https://github.com/UCSD-E4E/fishsense-lite/commit/5ec1ae99bf0bbd862ccc4cbd04006926ececfd72))
* **api:** dive_pipeline_status view for Superset dashboards ([f99372d](https://github.com/UCSD-E4E/fishsense-lite/commit/f99372d2122c04afec210a08b21708bd841665dd))

## [1.26.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.25.0...fishsense-api-v1.26.0) (2026-05-04)


### Features

* **headtail:** cascade stage 5.1 from valid laser labels ([56e46fe](https://github.com/UCSD-E4E/fishsense-lite/commit/56e46fec4598829702f55d8d3743e2aeb4a9a959))
* **headtail:** cascade stage 5.1 from valid laser labels ([2b73d28](https://github.com/UCSD-E4E/fishsense-lite/commit/2b73d2843e44c3df210b8fe6e71f16da2152610a))

## [1.25.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.24.1...fishsense-api-v1.25.0) (2026-05-03)


### Features

* **laser:** post-sync line-fit validation of laser labels (observe-only) ([9fa0e13](https://github.com/UCSD-E4E/fishsense-lite/commit/9fa0e130b83d14d8098b3758d8f9ab3f58f2460c))
* **laser:** post-sync line-fit validation of laser labels (observe-only) ([7390903](https://github.com/UCSD-E4E/fishsense-lite/commit/739090389c8a2e85ba99b34c687a98124982f827))

## [1.24.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.24.0...fishsense-api-v1.24.1) (2026-05-03)


### Bug Fixes

* **api,api-workflow-worker:** exclude NULL-project sentinels from cohort predicate ([5e09f1b](https://github.com/UCSD-E4E/fishsense-lite/commit/5e09f1bee9cb8f1512fc32253d5be58322118e08))
* exclude NULL-project sentinel rows from cohort predicate ([6380460](https://github.com/UCSD-E4E/fishsense-lite/commit/638046003a0ad62fc784bdff42b9588585905868))

## [1.24.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.23.0...fishsense-api-v1.24.0) (2026-05-03)


### Features

* **api-workflow-worker:** self-bootstrap populate + drop dives once labels seeded ([e03b7a6](https://github.com/UCSD-E4E/fishsense-lite/commit/e03b7a6ffd0730d4417f80b5e5b4b7e57f043d2e))
* **api-workflow-worker:** self-bootstrap populate + drop preprocessed dives ([a3609dd](https://github.com/UCSD-E4E/fishsense-lite/commit/a3609dd135ffce57964a0e0ef58031874f96084a))

## [1.23.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.22.2...fishsense-api-v1.23.0) (2026-05-02)


### Features

* **api-workflow-worker:** re-cohort stage 0.1 by labeling state + fix multi-row predicate ([aeed529](https://github.com/UCSD-E4E/fishsense-lite/commit/aeed529ab340168be644561e8c374ee1dadaeeeb))
* **api-workflow-worker:** re-cohort stage 0.1 by labeling state + fix multi-row predicate ([91a81c7](https://github.com/UCSD-E4E/fishsense-lite/commit/91a81c79386b2661470e2aad71585639824975f6))

## [1.22.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.22.1...fishsense-api-v1.22.2) (2026-05-02)


### Bug Fixes

* **lint:** silence pylint not-callable on func.count, drop unused import ([934f2a9](https://github.com/UCSD-E4E/fishsense-lite/commit/934f2a91c77a9666148ef86a787111fa11bb70c6))


### Performance Improvements

* **api-worker:** collapse dive cohort selectors to single SDK call ([62b6d13](https://github.com/UCSD-E4E/fishsense-lite/commit/62b6d1358a33a71ba6d15505d7ab05a4880e0b1a))
* **api-worker:** collapse dive cohort selectors to single SDK call ([fab3157](https://github.com/UCSD-E4E/fishsense-lite/commit/fab3157acf840f91b7a5ac7f73664ae178b64f4c))

## [1.22.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.22.0...fishsense-api-v1.22.1) (2026-05-02)


### Bug Fixes

* **api:** scope GET /dives/{id}/images/clusters/{ds} to data_source ([e311bc9](https://github.com/UCSD-E4E/fishsense-lite/commit/e311bc9511da61fe386130c8618cd7a26691eed7))
* **api:** scope GET /dives/{id}/images/clusters/{ds} to data_source ([85c4748](https://github.com/UCSD-E4E/fishsense-lite/commit/85c474853d0c328d381339e6f070df45630b349d))

## [1.22.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.21.0...fishsense-api-v1.22.0) (2026-05-01)


### Features

* **api-worker:** port stage 4.2 sync_species_labels ([91d17f4](https://github.com/UCSD-E4E/fishsense-lite/commit/91d17f44d7266fae2338fb6e3c8bef364940eb7b))

## [1.21.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.20.1...fishsense-api-v1.21.0) (2026-05-01)


### Features

* **api-worker:** port stage 12 sync_slate_label ([7922716](https://github.com/UCSD-E4E/fishsense-lite/commit/792271617a33bfc9c5907151814a02d3f21853ff))

## [1.20.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.20.0...fishsense-api-v1.20.1) (2026-05-01)


### Bug Fixes

* **api,sdk,api-workflow-worker:** single-call distinct-project-ids endpoint ([cef2d4d](https://github.com/UCSD-E4E/fishsense-lite/commit/cef2d4d1386d3d544680a9f49b6e78c75b596d7d))

## [1.20.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.19.0...fishsense-api-v1.20.0) (2026-05-01)


### Features

* **api-sdk:** generate wire-format models from OpenAPI schema ([d951aad](https://github.com/UCSD-E4E/fishsense-lite/commit/d951aad1731144e973ee70bf306ecd90b32330eb))
* **api,sdk:** incremental Label Studio sync via LabelStudioSyncCursor ([726a147](https://github.com/UCSD-E4E/fishsense-lite/commit/726a1477e436c8ca0235e60e2354e4154b1dac22))

## [1.19.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-v1.18.1...fishsense-api-v1.19.0) (2026-04-29)


### Features

* **ci:** build-once / promote-tag deploy split + monorepo-aware Dockerfiles ([66d20b9](https://github.com/UCSD-E4E/fishsense-lite/commit/66d20b9b9bc834bc2ebc88461006d71a7dd14faa))
* **ci:** root monorepo workflows (lint, release, docker) ([d58b806](https://github.com/UCSD-E4E/fishsense-lite/commit/d58b806b2ebf40e4826c7c601c8bd8ffd4fa4790))
* **shared:** extract fishsense-shared lib for Dynaconf, logging, TLS, ExceptionGroup helpers ([f896c5f](https://github.com/UCSD-E4E/fishsense-lite/commit/f896c5fc6017edc509e0e0c651da2b2a4c6519e6))


### Bug Fixes

* **api:** defer PG_CONNECTION_STRING construction to lifespan ([93b6111](https://github.com/UCSD-E4E/fishsense-lite/commit/93b6111288eee36a38941526c1649915859ba408))
* **lint:** pylint clean across the new worker code ([dbd832f](https://github.com/UCSD-E4E/fishsense-lite/commit/dbd832f61e062677f6477df9c38cf151162aee31))


### Documentation

* fill in package-level READMEs across services and libs ([df477db](https://github.com/UCSD-E4E/fishsense-lite/commit/df477dbb4c0956d4aa3864c66a2ffc13a31a9feb))

## v1.18.1 (2026-02-24)

### Bug Fixes

- Formatting of user controller
  ([`9fd0fa7`](https://github.com/UCSD-E4E/fishsense-api/commit/9fd0fa7b68bd5399161eadefa01aea0b4d35aae8))

### Chores

- **deps**: Bump pillow from 12.0.0 to 12.1.1
  ([`de1bd8f`](https://github.com/UCSD-E4E/fishsense-api/commit/de1bd8fe525182b693c8227d06b479aa2870a9ac))


## v1.18.0 (2026-02-24)

### Bug Fixes

- Use fastapi HTTPException in label_controller and redact email PII from logs
  ([`23c9a4d`](https://github.com/UCSD-E4E/fishsense-api/commit/23c9a4dae274d42879f679b90a103950475c3052))

### Features

- Add additional logging to user controller for 404 debugging
  ([`806cb2c`](https://github.com/UCSD-E4E/fishsense-api/commit/806cb2c5779032d0e49ca2876302a9bdcdf4a5b1))

- Add logging to all controllers for easier debugging
  ([`5d60e1b`](https://github.com/UCSD-E4E/fishsense-api/commit/5d60e1b7b1995e47f719f419a7342c568219536a))


## v1.17.0 (2026-02-24)

### Features

- Return 404s when user not found. add constraints to database
  ([`0f49aba`](https://github.com/UCSD-E4E/fishsense-api/commit/0f49aba9ed6c4d93d50acfb0d973a70c38f507cd))


## v1.16.0 (2026-02-12)

### Features

- Reintroduce more focused devcontainer
  ([`3ebbd06`](https://github.com/UCSD-E4E/fishsense-api/commit/3ebbd0692b7ed1712f7e271590d5e4c9b0e2724a))


## v1.15.3 (2026-02-08)

### Bug Fixes

- Pydantic warnings
  ([`a740fd6`](https://github.com/UCSD-E4E/fishsense-api/commit/a740fd6976d642f7859ba5b134261dae148bb285))

### Chores

- **deps**: Bump urllib3 from 2.5.0 to 2.6.3
  ([`3325fee`](https://github.com/UCSD-E4E/fishsense-api/commit/3325feeefe6baef0170d5b758124a6681954c9f0))


## v1.15.2 (2026-02-08)

### Bug Fixes

- Pylint errors
  ([`bd3ba38`](https://github.com/UCSD-E4E/fishsense-api/commit/bd3ba38f4e3acf6f114f8caf9a707d33b593f0e6))

- Reduce deadlocks for database
  ([`359f457`](https://github.com/UCSD-E4E/fishsense-api/commit/359f457619d7a6008888934771760dfe3c94a2a6))


## v1.15.1 (2026-01-05)

### Bug Fixes

- Use the correct deserializer for datetimes in api
  ([`789ac28`](https://github.com/UCSD-E4E/fishsense-api/commit/789ac28cccf3cbbc10bb74694bece8247175b30a))


## v1.15.0 (2025-12-19)

### Features

- Add endpoint to grab headtail label by label studio id
  ([`64a5b9a`](https://github.com/UCSD-E4E/fishsense-api/commit/64a5b9a71388d3212f5e7b43c91581b1de901437))


## v1.14.2 (2025-12-08)

### Bug Fixes

- Not returning proper groups
  ([`5583827`](https://github.com/UCSD-E4E/fishsense-api/commit/5583827552feb092d5011890c6ba3127856e2aa8))


## v1.14.1 (2025-12-08)

### Bug Fixes

- 500 error in clustering
  ([`b496e7c`](https://github.com/UCSD-E4E/fishsense-api/commit/b496e7ce9ee4c181f8353f46845fc3ddf7ae7f19))


## v1.14.0 (2025-12-03)

### Bug Fixes

- Properly set the version number
  ([`c9c9574`](https://github.com/UCSD-E4E/fishsense-api/commit/c9c9574fed6869492f5c28c7a85f5b56528573c8))

### Features

- Specify version module for semantic release
  ([`ce46b15`](https://github.com/UCSD-E4E/fishsense-api/commit/ce46b1548aae8fc4275fa1ec7dc692f602f354e1))


## v1.13.0 (2025-12-02)

### Features

- Allow getting user by label studio id
  ([`33602fa`](https://github.com/UCSD-E4E/fishsense-api/commit/33602fa06baf44787dbee6480fd71b073c96a349))


## v1.12.2 (2025-12-02)

### Bug Fixes

- Label studio id should be an int
  ([`b6ca428`](https://github.com/UCSD-E4E/fishsense-api/commit/b6ca4282ccd0a8c5dc4f21e390b43266639c4108))


## v1.12.1 (2025-12-02)

### Bug Fixes

- Typo in laser label task id
  ([`7cc7aca`](https://github.com/UCSD-E4E/fishsense-api/commit/7cc7acad435ec48597727670b5447873253a8c46))


## v1.12.0 (2025-12-02)

### Features

- Add method for laser label by label studio id
  ([`04d4b20`](https://github.com/UCSD-E4E/fishsense-api/commit/04d4b20ba7bf677eea4090c1a723bbfb9c363dbb))


## v1.11.0 (2025-11-30)

### Bug Fixes

- Pylint errors
  ([`3e558da`](https://github.com/UCSD-E4E/fishsense-api/commit/3e558daa3d32416a869b4670af6d2e734e2d62b3))

### Features

- Be able to measure fish
  ([`ec161b4`](https://github.com/UCSD-E4E/fishsense-api/commit/ec161b4cbd09b462c02c79eb38fb7135e0320cd3))

- Create final groupings
  ([`91d12c5`](https://github.com/UCSD-E4E/fishsense-api/commit/91d12c54e8480110f7dab36a7ef3d7e455c91f75))

- Introduce laser extrinsics
  ([`2961e80`](https://github.com/UCSD-E4E/fishsense-api/commit/2961e80beae39bfb3db0a292c676913c452fc740))


## v1.10.0 (2025-11-24)

### Features

- Dive slate labels synced
  ([`d157ac1`](https://github.com/UCSD-E4E/fishsense-api/commit/d157ac174c97da1085791df8ea9e12bb6df54c84))


## v1.9.0 (2025-11-24)

### Bug Fixes

- Pylint errors
  ([`a5addfb`](https://github.com/UCSD-E4E/fishsense-api/commit/a5addfb73b84217b0a5c043b1cd391d2909b7864))

### Features

- Add dive slate labels
  ([`75cc762`](https://github.com/UCSD-E4E/fishsense-api/commit/75cc7621a7b3ea8f4e32ebd665fbfb0410d85a57))


## v1.8.0 (2025-11-23)

### Chores

- Pylint errors
  ([`b0353b4`](https://github.com/UCSD-E4E/fishsense-api/commit/b0353b4f75af2458af2510f932d805226f16dc90))

### Features

- Sync headtail labels
  ([`d9d8541`](https://github.com/UCSD-E4E/fishsense-api/commit/d9d8541ef19d1063bee25d39ae06941264e54458))


## v1.7.0 (2025-11-23)

### Bug Fixes

- Pylint errors
  ([`0f5b7eb`](https://github.com/UCSD-E4E/fishsense-api/commit/0f5b7eb87abd6e764354ce223ff47f8b5b316cec))

### Features

- Expose laser label
  ([`84ba9b8`](https://github.com/UCSD-E4E/fishsense-api/commit/84ba9b8ebb7255d7589972821fb4f56d7b62b69b))


## v1.6.0 (2025-11-19)

### Features

- Full species labels
  ([`129e4cd`](https://github.com/UCSD-E4E/fishsense-api/commit/129e4cdb872adfd33f8b9d124b8614961a4759bb))


## v1.5.0 (2025-11-18)

### Features

- Introduce user controller
  ([`4a66b2d`](https://github.com/UCSD-E4E/fishsense-api/commit/4a66b2d84e6fb07bffbc6c3b6e1f722c8ae3fb8a))


## v1.4.0 (2025-11-17)

### Features

- Add data frame cluster datasource
  ([`cc80c4f`](https://github.com/UCSD-E4E/fishsense-api/commit/cc80c4f0872925b31dde38ee0cff9e3491ab58b5))


## v1.3.0 (2025-11-17)

### Features

- Add project id to species labels
  ([`08ae468`](https://github.com/UCSD-E4E/fishsense-api/commit/08ae468e221b154fee39b7958365853e1485dcd9))


## v1.2.0 (2025-11-04)

### Features

- Introduce label studio species project id to sql table
  ([`bda896b`](https://github.com/UCSD-E4E/fishsense-api/commit/bda896b5d9f5269e1025ed3160987cc64bcf0e88))


## v1.1.1 (2025-10-30)

### Bug Fixes

- Remove flakes readonly file system issues with notebooks
  ([`037b93d`](https://github.com/UCSD-E4E/fishsense-api/commit/037b93d06a7f20ac8de12186b50f18e7fda83575))


## v1.1.0 (2025-10-30)

### Features

- Introduce code owners
  ([`e4d4244`](https://github.com/UCSD-E4E/fishsense-api/commit/e4d42447e11bd1ce7ab41d88dcc47ec9040d4687))


## v1.0.5 (2025-10-30)

### Bug Fixes

- Cleanup settings to not have user included
  ([`4b8e999`](https://github.com/UCSD-E4E/fishsense-api/commit/4b8e999facb6b1469ccebf62f4c517e53cfc3eb7))

### Chores

- Update uv.lock
  ([`2f6857c`](https://github.com/UCSD-E4E/fishsense-api/commit/2f6857cf422e723353162141d838f4fb8c14d980))


## v1.0.4 (2025-10-29)

### Bug Fixes

- Check in missing parts of direnv
  ([`7997e2d`](https://github.com/UCSD-E4E/fishsense-api/commit/7997e2ddb656bf09f30964bb2b9858511422dc4c))


## v1.0.3 (2025-10-29)

### Bug Fixes

- Introduce flakes
  ([`b597faf`](https://github.com/UCSD-E4E/fishsense-api/commit/b597faf09b1bbb62af2b3ae9032d7344f913da10))


## v1.0.2 (2025-10-27)

### Bug Fixes

- Pylint errors
  ([`b43cb66`](https://github.com/UCSD-E4E/fishsense-api/commit/b43cb6601b5b0bd2a01345e1f6e9b9170b71b51b))

- Use a connection pool so that we do not flood the database with connections
  ([`3086756`](https://github.com/UCSD-E4E/fishsense-api/commit/3086756e451e77944e006f54b261fe42fdcf9d07))

### Chores

- Update uv lock
  ([`2c591f0`](https://github.com/UCSD-E4E/fishsense-api/commit/2c591f007183dd764d54b1bd2177a41f91421419))


## v1.0.1 (2025-10-21)

### Bug Fixes

- Fishsense api server description
  ([`0b73e00`](https://github.com/UCSD-E4E/fishsense-api/commit/0b73e0046725cedebddc788a868da688ef2a234e))


## v1.0.0 (2025-10-21)

- Initial Release
