# Changelog

## [0.7.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.7.0...fishsense-shared-v0.7.1) (2026-08-08)


### Bug Fixes

* **api:** create views on a fresh DB; Postgres integration tests; coverage ratchet ([#550](https://github.com/UCSD-E4E/fishsense-lite/issues/550)) ([dbdda9e](https://github.com/UCSD-E4E/fishsense-lite/commit/dbdda9eb4acc99d35125909dccb2c413c402fdf4))
* **ci:** run every package's tests; fix E4EFS_DOCKER; cover the portal auth gate ([#546](https://github.com/UCSD-E4E/fishsense-lite/issues/546)) ([e3d01b5](https://github.com/UCSD-E4E/fishsense-lite/commit/e3d01b5419b5dc01452e7907fe13c101932e0942))

## [0.7.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.6.0...fishsense-shared-v0.7.0) (2026-08-02)


### Features

* **slate:** slate-predict parent workflow + resolver + persist + schedule (part 3b) ([#484](https://github.com/UCSD-E4E/fishsense-lite/issues/484)) ([b297888](https://github.com/UCSD-E4E/fishsense-lite/commit/b297888620e825426da382a9af6470352336d7af))

## [0.6.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.5.0...fishsense-shared-v0.6.0) (2026-07-28)


### Features

* **laser-detector:** deps + GPU (phase 2) ([#420](https://github.com/UCSD-E4E/fishsense-lite/issues/420)) ([fdddfe8](https://github.com/UCSD-E4E/fishsense-lite/commit/fdddfe8aabc6ee040617672c1f03d6adc6472e23))
* **laser-detector:** predict activity + fan-out workflow (phase 1) ([#418](https://github.com/UCSD-E4E/fishsense-lite/issues/418)) ([e30d36f](https://github.com/UCSD-E4E/fishsense-lite/commit/e30d36f1e606a42aab6fbdd7266ddfd2eb6d511c))
* **laser-populate:** assisted-review pre-annotations, decoupled (phase 5) ([#423](https://github.com/UCSD-E4E/fishsense-lite/issues/423)) ([1af7aa9](https://github.com/UCSD-E4E/fishsense-lite/commit/1af7aa90dc9895bd8e98fa77823a13676882f789))
* **laser-predict:** orchestration — parent + selector + schedule (phase 4) ([#422](https://github.com/UCSD-E4E/fishsense-lite/issues/422)) ([28baa3c](https://github.com/UCSD-E4E/fishsense-lite/commit/28baa3cc22ee9da02344f4901f9bb83fb20c1642))

## [0.5.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.4.1...fishsense-shared-v0.5.0) (2026-07-15)


### Features

* **temporal:** connect all workers to the `fishsense` namespace ([#266](https://github.com/UCSD-E4E/fishsense-lite/issues/266)) ([10a2823](https://github.com/UCSD-E4E/fishsense-lite/commit/10a28232378f862f855ec4ef2fd0db5354c0882d))

## [0.4.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.4.0...fishsense-shared-v0.4.1) (2026-05-07)


### Bug Fixes

* **api-workflow-worker:** suppress validation-pass failures in laser sync ([88b9404](https://github.com/UCSD-E4E/fishsense-lite/commit/88b9404c499a36eec5d3f0650ce9b70078d7b2d0))
* **api-workflow-worker:** suppress validation-pass failures in laser sync ([79faf2e](https://github.com/UCSD-E4E/fishsense-lite/commit/79faf2e4b7a30225122093687f4017eaa777d57a))
* **fishsense-shared:** only suppress Exception subclasses, never BaseException ([746dbca](https://github.com/UCSD-E4E/fishsense-lite/commit/746dbca9e5ffc5a6ebbd20b67d2f27b8b445950e))

## [0.4.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.3.0...fishsense-shared-v0.4.0) (2026-05-06)


### Features

* species cascade flip + stage 1 clustering automation ([ec565f5](https://github.com/UCSD-E4E/fishsense-lite/commit/ec565f5ffe4c53c24196c69ff3a2554098034db9))

## [0.3.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.2.1...fishsense-shared-v0.3.0) (2026-05-02)


### Features

* **api-worker:** apply parent/child pattern to stages 2, 5.1, 9 ([a22c5c5](https://github.com/UCSD-E4E/fishsense-lite/commit/a22c5c5988c8d2c2cf9e7fe0f57d3b466ca25500))
* **api-worker:** stage 0.1 parent + cluster-safe schedule ([504b05f](https://github.com/UCSD-E4E/fishsense-lite/commit/504b05f03e489abda424aabf0fdef5043c669346))

## [0.2.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.2.0...fishsense-shared-v0.2.1) (2026-05-01)


### Documentation

* refresh CLAUDE.md, diagrams, and READMEs for new collaborators ([e45db82](https://github.com/UCSD-E4E/fishsense-lite/commit/e45db82467b3b92300c9af767cab7f4331e4d4d6))

## [0.2.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-shared-v0.1.0...fishsense-shared-v0.2.0) (2026-04-29)


### Features

* **shared:** extract fishsense-shared lib for Dynaconf, logging, TLS, ExceptionGroup helpers ([f896c5f](https://github.com/UCSD-E4E/fishsense-lite/commit/f896c5fc6017edc509e0e0c651da2b2a4c6519e6))


### Documentation

* fill in package-level READMEs across services and libs ([df477db](https://github.com/UCSD-E4E/fishsense-lite/commit/df477dbb4c0956d4aa3864c66a2ffc13a31a9feb))
