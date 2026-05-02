# Changelog

## [2.0.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v0.4.0...fishsense-data-processing-workflow-worker-v2.0.0) (2026-05-02)


### Miscellaneous Chores

* **data-worker:** cut 2.0.0 release ([9782505](https://github.com/UCSD-E4E/fishsense-lite/commit/978250542757d0d86ce2c9c07a5a54c91464eeb1))

## [0.4.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v0.3.0...fishsense-data-processing-workflow-worker-v0.4.0) (2026-05-02)


### Features

* **api-worker:** apply parent/child pattern to stages 2, 5.1, 9 ([a22c5c5](https://github.com/UCSD-E4E/fishsense-lite/commit/a22c5c5988c8d2c2cf9e7fe0f57d3b466ca25500))
* **api-worker:** stage 0.1 parent + cluster-safe schedule ([504b05f](https://github.com/UCSD-E4E/fishsense-lite/commit/504b05f03e489abda424aabf0fdef5043c669346))
* **data-worker:** self-pacing hourly stage 0.1 with HIGH-priority selector ([2a1b0f5](https://github.com/UCSD-E4E/fishsense-lite/commit/2a1b0f5b3efe301e86c0ca628f097644519068c1))


### Bug Fixes

* **deploy:** drop orphan laser_jpeg nginx rewrite + clarify file-exchange folder names ([f039232](https://github.com/UCSD-E4E/fishsense-lite/commit/f03923237f66b7f0bf4606902611031ae9002b45))

## [0.3.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v0.2.0...fishsense-data-processing-workflow-worker-v0.3.0) (2026-05-01)


### Features

* **data-worker:** port stage 13 perform_laser_calibration ([fdac76b](https://github.com/UCSD-E4E/fishsense-lite/commit/fdac76b275d0108cd80d9f965fa2659915776f33))
* **data-worker:** port stage 14 measure_fish ([0283023](https://github.com/UCSD-E4E/fishsense-lite/commit/02830230259fd033b44b8b86e0f0ed3a8ddf3d84))

## [0.2.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v0.1.0...fishsense-data-processing-workflow-worker-v0.2.0) (2026-04-29)


### Features

* **ci:** build-once / promote-tag deploy split + monorepo-aware Dockerfiles ([66d20b9](https://github.com/UCSD-E4E/fishsense-lite/commit/66d20b9b9bc834bc2ebc88461006d71a7dd14faa))
* **data-worker:** gate new preprocess workflows behind feature flag ([fd261e4](https://github.com/UCSD-E4E/fishsense-lite/commit/fd261e472710b613d09b5dad01e1324e25e496a7))
* **data-worker:** TDD port of stage 0.1 preprocess_laser_images ([2d1f2dc](https://github.com/UCSD-E4E/fishsense-lite/commit/2d1f2dc0e1845d69f6cf884f55068fa6b006c516))
* **data-worker:** TDD port of stage 9 preprocess_slate_images ([e738ce4](https://github.com/UCSD-E4E/fishsense-lite/commit/e738ce4590cd792fb5303b31041febb6abe2a0b2))
* **data-worker:** TDD port of stage2 preprocess_dive_images ([669f933](https://github.com/UCSD-E4E/fishsense-lite/commit/669f9338357665955b8e17038d0be0f14b56ddc2))
* **shared:** extract fishsense-shared lib for Dynaconf, logging, TLS, ExceptionGroup helpers ([f896c5f](https://github.com/UCSD-E4E/fishsense-lite/commit/f896c5fc6017edc509e0e0c651da2b2a4c6519e6))


### Bug Fixes

* **data-processing-worker:** rename models_tmp.py to models.py ([9be8aff](https://github.com/UCSD-E4E/fishsense-lite/commit/9be8aff73dfe48e7fa6272a32e4f4662fc775f5c))
* **lint:** pylint clean across the new worker code ([dbd832f](https://github.com/UCSD-E4E/fishsense-lite/commit/dbd832f61e062677f6477df9c38cf151162aee31))


### Documentation

* fill in package-level READMEs across services and libs ([df477db](https://github.com/UCSD-E4E/fishsense-lite/commit/df477dbb4c0956d4aa3864c66a2ffc13a31a9feb))
