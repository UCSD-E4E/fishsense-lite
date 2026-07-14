# Changelog

## [2.6.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.5.0...fishsense-data-processing-workflow-worker-v2.6.0) (2026-07-14)


### Features

* **labels:** give SpeciesLabel + DiveSlateLabel a `superseded` dead-letter flag ([#247](https://github.com/UCSD-E4E/fishsense-lite/issues/247)) ([f851e93](https://github.com/UCSD-E4E/fishsense-lite/commit/f851e933d4e3b2c8f596b945d43b9fd0124241b6))

## [2.5.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.4.1...fishsense-data-processing-workflow-worker-v2.5.0) (2026-07-10)


### Features

* **data-worker:** move to NRP/Kubernetes (scale-to-zero) on the Garage base ([942a57e](https://github.com/UCSD-E4E/fishsense-lite/commit/942a57e9a68f2fdfd625c2b55fb592fdf2795f1f))
* **data-worker:** NRP/Kubernetes scale-to-zero (stacked on Garage) ([1c76411](https://github.com/UCSD-E4E/fishsense-lite/commit/1c76411445db20c8a4322007de376d223af57c34))
* migrate worker storage from nginx file-exchange to Garage object store ([34bb385](https://github.com/UCSD-E4E/fishsense-lite/commit/34bb385595732f77b4a01156a16e2c5515339783))
* migrate worker storage from nginx file-exchange to Garage object store ([88e88f9](https://github.com/UCSD-E4E/fishsense-lite/commit/88e88f9df6df83768d831e3ea0e6e8d6360a2130))


### Bug Fixes

* **ci:** integration — LS S3 endpoint must be reachable from the LS container ([#228](https://github.com/UCSD-E4E/fishsense-lite/issues/228)) ([1a3a2f3](https://github.com/UCSD-E4E/fishsense-lite/commit/1a3a2f35e85c73be1e259225342eaac0489d4618))
* **object-store:** address Copilot review on PR [#205](https://github.com/UCSD-E4E/fishsense-lite/issues/205) ([439958f](https://github.com/UCSD-E4E/fishsense-lite/commit/439958fa1c70f13bad6bf4043967ee80e4b15836))

## [2.4.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.4.0...fishsense-data-processing-workflow-worker-v2.4.1) (2026-05-06)


### Bug Fixes

* **consumers:** pin fishsense-api-sdk&gt;=1.35.0 ([f12c6d0](https://github.com/UCSD-E4E/fishsense-lite/commit/f12c6d0b876df981fe85d4be247be878e9d2fdd1))
* **deps:** pin fishsense-api-sdk&gt;=1.35.0 ([04f9e2a](https://github.com/UCSD-E4E/fishsense-lite/commit/04f9e2ab0d3319ba44a4bd520927c50dd42f1f1d))

## [2.4.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.3.1...fishsense-data-processing-workflow-worker-v2.4.0) (2026-05-06)


### Features

* species cascade flip + stage 1 clustering automation ([ec565f5](https://github.com/UCSD-E4E/fishsense-lite/commit/ec565f5ffe4c53c24196c69ff3a2554098034db9))

## [2.3.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.3.0...fishsense-data-processing-workflow-worker-v2.3.1) (2026-05-05)


### Bug Fixes

* **data-worker:** silence pylint unused-argument on timeout-test stubs ([5f3b81b](https://github.com/UCSD-E4E/fishsense-lite/commit/5f3b81b4ef9f7eef5a254aaf02d4bf22edd2c2ee))
* **data-worker:** use start_to_close on per-image preprocess activities ([4bea338](https://github.com/UCSD-E4E/fishsense-lite/commit/4bea33863731cbcf598d4d6f04fa83dc84ad7b83))
* **data-worker:** use start_to_close on per-image preprocess activities ([1fe1193](https://github.com/UCSD-E4E/fishsense-lite/commit/1fe11938d9ebe047bc6b1b7779455b31c53a7285))

## [2.3.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.2.4...fishsense-data-processing-workflow-worker-v2.3.0) (2026-05-05)


### Features

* **workers:** heartbeat activity loops for progress visibility ([d19771f](https://github.com/UCSD-E4E/fishsense-lite/commit/d19771fcbc475a167a4a772ec5f57b133394b971))
* **workers:** heartbeat activity loops for progress visibility ([c6e9448](https://github.com/UCSD-E4E/fishsense-lite/commit/c6e94480d35c1e8d42318332d762e097368d1b6e))

## [2.2.4](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.2.3...fishsense-data-processing-workflow-worker-v2.2.4) (2026-05-04)


### Bug Fixes

* **deps:** pin fishsense-api-sdk&gt;=1.34.1 + auto-cascade SDK releases ([9437a52](https://github.com/UCSD-E4E/fishsense-lite/commit/9437a52633ed91e92b82da4eeaa3a12d5aa1be9a))
* **deps:** pin fishsense-api-sdk&gt;=1.34.1 + auto-cascade SDK releases ([d364361](https://github.com/UCSD-E4E/fishsense-lite/commit/d3643614403321a330728321d8b6119d572b1f3d))

## [2.2.3](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.2.2...fishsense-data-processing-workflow-worker-v2.2.3) (2026-05-04)


### Bug Fixes

* **laser:** refuse to supersede when outlier fraction &gt; 50% ([5e7ade3](https://github.com/UCSD-E4E/fishsense-lite/commit/5e7ade3cfc2e0973dcbf92aeb0171b588931f8d1))
* **laser:** refuse to supersede when outlier fraction &gt; 50% ([6af9ea5](https://github.com/UCSD-E4E/fishsense-lite/commit/6af9ea5069f0c453413d7b89775678876f96b81d))

## [2.2.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.2.1...fishsense-data-processing-workflow-worker-v2.2.2) (2026-05-04)


### Bug Fixes

* **laser:** pump heartbeats during slow get_laser_labels ([e013c58](https://github.com/UCSD-E4E/fishsense-lite/commit/e013c58cb5aac8cfb5fc175a1b130c039f2235c4))
* **laser:** pump heartbeats during slow get_laser_labels ([bd04dc0](https://github.com/UCSD-E4E/fishsense-lite/commit/bd04dc04276d796eec9bdee6b33ff09290d18d73))

## [2.2.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.2.0...fishsense-data-processing-workflow-worker-v2.2.1) (2026-05-03)


### Bug Fixes

* **laser:** supersede PUTs run concurrently, capped at 8 ([4bfaed0](https://github.com/UCSD-E4E/fishsense-lite/commit/4bfaed095bdd56bcf37fff0932125e5cb5c57e52))
* **laser:** supersede PUTs run concurrently, capped at 8 ([4ba4716](https://github.com/UCSD-E4E/fishsense-lite/commit/4ba47165a573387d1196867eb86139bab2a4089a))

## [2.2.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.1.1...fishsense-data-processing-workflow-worker-v2.2.0) (2026-05-03)


### Features

* **laser:** supersede flagged laser labels (Phase 2) ([e6064f1](https://github.com/UCSD-E4E/fishsense-lite/commit/e6064f15e7f553cb6970b30d4d0a5566c2fc35ff))
* **laser:** supersede flagged laser labels (Phase 2) ([e5c44e6](https://github.com/UCSD-E4E/fishsense-lite/commit/e5c44e6b09cbd01189add1f53341530d18282e7c))

## [2.1.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.1.0...fishsense-data-processing-workflow-worker-v2.1.1) (2026-05-03)


### Bug Fixes

* **laser:** instrument + bump timeouts on validate-laser-labels activity ([570d10f](https://github.com/UCSD-E4E/fishsense-lite/commit/570d10f59b090b6d27c3e7a83c1dc6e8fa580c5c))
* **laser:** timeouts + heartbeats on validate activity; .secrets.toml.example for all workers ([65149fa](https://github.com/UCSD-E4E/fishsense-lite/commit/65149fac706acaf20f8e88e26337e40fcdd9892a))

## [2.1.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-data-processing-workflow-worker-v2.0.0...fishsense-data-processing-workflow-worker-v2.1.0) (2026-05-03)


### Features

* **laser:** post-sync line-fit validation of laser labels (observe-only) ([9fa0e13](https://github.com/UCSD-E4E/fishsense-lite/commit/9fa0e130b83d14d8098b3758d8f9ab3f58f2460c))
* **laser:** post-sync line-fit validation of laser labels (observe-only) ([7390903](https://github.com/UCSD-E4E/fishsense-lite/commit/739090389c8a2e85ba99b34c687a98124982f827))


### Bug Fixes

* **laser:** satisfy pylint on the validation module + tests ([a21fcd4](https://github.com/UCSD-E4E/fishsense-lite/commit/a21fcd451a5aab43dca3b3f0a84df6afe17dba31))

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
