# CHANGELOG

<!-- version list -->

## [1.41.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.41.0...fishsense-api-workflow-worker-v1.41.1) (2026-07-18)


### Bug Fixes

* **deps:** pin fishsense-api-sdk&gt;=1.37.0 ([#311](https://github.com/UCSD-E4E/fishsense-lite/issues/311)) ([6de358c](https://github.com/UCSD-E4E/fishsense-lite/commit/6de358c672b14472684481a41f3b5c31b75895fb))

## [1.41.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.40.1...fishsense-api-workflow-worker-v1.41.0) (2026-07-18)


### Features

* **populate-species:** make species populate idempotent for scheduling ([#308](https://github.com/UCSD-E4E/fishsense-lite/issues/308)) ([fa76220](https://github.com/UCSD-E4E/fishsense-lite/commit/fa7622030f2771c8ca956bd11673ac952b4d9a49))
* **populate-species:** scheduled populate parent (decoupled, superseded-aware) ([#309](https://github.com/UCSD-E4E/fishsense-lite/issues/309)) ([082b27d](https://github.com/UCSD-E4E/fishsense-lite/commit/082b27de33ab7ccc90081d0daefa92b8596afa20))


### Bug Fixes

* **object-store:** repoint Garage endpoint to s3.e4e.ucsd.edu ([#307](https://github.com/UCSD-E4E/fishsense-lite/issues/307)) ([07b3ea4](https://github.com/UCSD-E4E/fishsense-lite/commit/07b3ea48a709b0b6d15f540a38c808ffe6dda98e))

## [1.40.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.40.0...fishsense-api-workflow-worker-v1.40.1) (2026-07-17)


### Bug Fixes

* **api-worker:** wake NRP data-worker before laser-label validation fan-out ([#303](https://github.com/UCSD-E4E/fishsense-lite/issues/303)) ([354a75f](https://github.com/UCSD-E4E/fishsense-lite/commit/354a75f4f9c997b0c06338fe3d7e5dea44e44cac))

## [1.40.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.39.0...fishsense-api-workflow-worker-v1.40.0) (2026-07-17)


### Features

* **stage14:** schedule fish measurement hourly at +40 ([#293](https://github.com/UCSD-E4E/fishsense-lite/issues/293)) ([574c44f](https://github.com/UCSD-E4E/fishsense-lite/commit/574c44f3a5ac64ec5036aca5c783dfd658cbe4f8))


### Bug Fixes

* **deps:** pin fishsense-api-sdk&gt;=1.36.1 ([#289](https://github.com/UCSD-E4E/fishsense-lite/issues/289)) ([45f73ac](https://github.com/UCSD-E4E/fishsense-lite/commit/45f73ac7e7f915b636c0e038fe56f0d633a86665))

## [1.39.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.38.1...fishsense-api-workflow-worker-v1.39.0) (2026-07-16)


### Features

* **labels:** create per-dive LS projects in a configured workspace ([#276](https://github.com/UCSD-E4E/fishsense-lite/issues/276)) ([47fdf02](https://github.com/UCSD-E4E/fishsense-lite/commit/47fdf02229cff43b79c08e60a5668b2d2e18aee8))

## [1.38.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.38.0...fishsense-api-workflow-worker-v1.38.1) (2026-07-15)


### Bug Fixes

* **scaling:** verify NRP apiserver TLS without OpenSSL 3.x strict mode ([#273](https://github.com/UCSD-E4E/fishsense-lite/issues/273)) ([8b23a64](https://github.com/UCSD-E4E/fishsense-lite/commit/8b23a645ab745138cc89c28291c086fba24cd77f))

## [1.38.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.37.1...fishsense-api-workflow-worker-v1.38.0) (2026-07-15)


### Features

* **temporal:** connect all workers to the `fishsense` namespace ([#266](https://github.com/UCSD-E4E/fishsense-lite/issues/266)) ([10a2823](https://github.com/UCSD-E4E/fishsense-lite/commit/10a28232378f862f855ec4ef2fd0db5354c0882d))

## [1.37.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.37.0...fishsense-api-workflow-worker-v1.37.1) (2026-07-15)


### Bug Fixes

* **deps:** pin fishsense-api-sdk&gt;=1.36.0 ([#260](https://github.com/UCSD-E4E/fishsense-lite/issues/260)) ([d6e0a50](https://github.com/UCSD-E4E/fishsense-lite/commit/d6e0a5007c6710566949994ca545f689ab51d275))

## [1.37.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.36.0...fishsense-api-workflow-worker-v1.37.0) (2026-07-14)


### Features

* **labels:** give SpeciesLabel + DiveSlateLabel a `superseded` dead-letter flag ([#247](https://github.com/UCSD-E4E/fishsense-lite/issues/247)) ([f851e93](https://github.com/UCSD-E4E/fishsense-lite/commit/f851e933d4e3b2c8f596b945d43b9fd0124241b6))


### Bug Fixes

* **worker:** label sync tolerates an unmapped annotator (don't crash) ([#249](https://github.com/UCSD-E4E/fishsense-lite/issues/249)) ([c446d14](https://github.com/UCSD-E4E/fishsense-lite/commit/c446d14ab9f3685d3570024bac0324b62db157a0))

## [1.36.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.35.3...fishsense-api-workflow-worker-v1.36.0) (2026-07-10)


### Features

* **data-worker:** move to NRP/Kubernetes (scale-to-zero) on the Garage base ([942a57e](https://github.com/UCSD-E4E/fishsense-lite/commit/942a57e9a68f2fdfd625c2b55fb592fdf2795f1f))
* **data-worker:** NRP/Kubernetes scale-to-zero (stacked on Garage) ([1c76411](https://github.com/UCSD-E4E/fishsense-lite/commit/1c76411445db20c8a4322007de376d223af57c34))
* migrate worker storage from nginx file-exchange to Garage object store ([34bb385](https://github.com/UCSD-E4E/fishsense-lite/commit/34bb385595732f77b4a01156a16e2c5515339783))
* migrate worker storage from nginx file-exchange to Garage object store ([88e88f9](https://github.com/UCSD-E4E/fishsense-lite/commit/88e88f9df6df83768d831e3ea0e6e8d6360a2130))


### Bug Fixes

* address Copilot review on PR [#207](https://github.com/UCSD-E4E/fishsense-lite/issues/207) ([4f49cad](https://github.com/UCSD-E4E/fishsense-lite/commit/4f49cad19e2ebc43079f9df0eb0308c92c5a14c1))
* **object-store:** address Copilot review on PR [#205](https://github.com/UCSD-E4E/fishsense-lite/issues/205) ([439958f](https://github.com/UCSD-E4E/fishsense-lite/commit/439958fa1c70f13bad6bf4043967ee80e4b15836))
* **test:** api-worker integration tests must use the real Garage endpoint ([dbe4155](https://github.com/UCSD-E4E/fishsense-lite/commit/dbe4155db8c032b6e94f5f8623c30c6c840cd237))

## [1.35.3](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.35.2...fishsense-api-workflow-worker-v1.35.3) (2026-05-07)


### Bug Fixes

* **api-workflow-worker:** suppress validation-pass failures in laser sync ([88b9404](https://github.com/UCSD-E4E/fishsense-lite/commit/88b9404c499a36eec5d3f0650ce9b70078d7b2d0))
* **api-workflow-worker:** suppress validation-pass failures in laser sync ([79faf2e](https://github.com/UCSD-E4E/fishsense-lite/commit/79faf2e4b7a30225122093687f4017eaa777d57a))

## [1.35.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.35.1...fishsense-api-workflow-worker-v1.35.2) (2026-05-07)


### Bug Fixes

* **api-workflow-worker:** address pylint + Copilot review feedback ([9a0255a](https://github.com/UCSD-E4E/fishsense-lite/commit/9a0255a9ba48e0af72a27fd299c27b1745dc098a))
* **api-workflow-worker:** replace synology-api with synology-filestation ([5ee188d](https://github.com/UCSD-E4E/fishsense-lite/commit/5ee188d8e68caf35dfdbdee8cfa95dee52923744))
* **api-workflow-worker:** replace synology-api with synology-filestation ([3821818](https://github.com/UCSD-E4E/fishsense-lite/commit/3821818103de07ad8197f1e0da952099415ebdf8))

## [1.35.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.35.0...fishsense-api-workflow-worker-v1.35.1) (2026-05-06)


### Bug Fixes

* **consumers:** pin fishsense-api-sdk&gt;=1.35.0 ([f12c6d0](https://github.com/UCSD-E4E/fishsense-lite/commit/f12c6d0b876df981fe85d4be247be878e9d2fdd1))
* **deps:** pin fishsense-api-sdk&gt;=1.35.0 ([04f9e2a](https://github.com/UCSD-E4E/fishsense-lite/commit/04f9e2ab0d3319ba44a4bd520927c50dd42f1f1d))

## [1.35.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.34.0...fishsense-api-workflow-worker-v1.35.0) (2026-05-06)


### Features

* species cascade flip + stage 1 clustering automation ([ec565f5](https://github.com/UCSD-E4E/fishsense-lite/commit/ec565f5ffe4c53c24196c69ff3a2554098034db9))

## [1.34.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.33.0...fishsense-api-workflow-worker-v1.34.0) (2026-05-05)


### Features

* **workers:** heartbeat activity loops for progress visibility ([d19771f](https://github.com/UCSD-E4E/fishsense-lite/commit/d19771fcbc475a167a4a772ec5f57b133394b971))
* **workers:** heartbeat activity loops for progress visibility ([c6e9448](https://github.com/UCSD-E4E/fishsense-lite/commit/c6e94480d35c1e8d42318332d762e097368d1b6e))

## [1.33.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.32.1...fishsense-api-workflow-worker-v1.33.0) (2026-05-05)


### Features

* **api-worker:** per-dive Label Studio projects (one project per dive per stage) ([2cff661](https://github.com/UCSD-E4E/fishsense-lite/commit/2cff6613ada94d8a118fd0723d93db0d024388fe))


### Bug Fixes

* **api-worker:** cap per-dive LS project titles at LS's 50-char limit ([246215c](https://github.com/UCSD-E4E/fishsense-lite/commit/246215c81460af25a57b61943eb247d24ab4cfba))
* **api-worker:** make _patch_dive_lookup stub accept kwargs ([b05deef](https://github.com/UCSD-E4E/fishsense-lite/commit/b05deef34a7effb3eb896473b06d4beafe83cba8))

## [1.32.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.32.0...fishsense-api-workflow-worker-v1.32.1) (2026-05-04)


### Bug Fixes

* **deps:** pin fishsense-api-sdk&gt;=1.34.1 + auto-cascade SDK releases ([9437a52](https://github.com/UCSD-E4E/fishsense-lite/commit/9437a52633ed91e92b82da4eeaa3a12d5aa1be9a))
* **deps:** pin fishsense-api-sdk&gt;=1.34.1 + auto-cascade SDK releases ([d364361](https://github.com/UCSD-E4E/fishsense-lite/commit/d3643614403321a330728321d8b6119d572b1f3d))

## [1.32.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.31.1...fishsense-api-workflow-worker-v1.32.0) (2026-05-04)


### Features

* **headtail:** cascade stage 5.1 from valid laser labels ([56e46fe](https://github.com/UCSD-E4E/fishsense-lite/commit/56e46fec4598829702f55d8d3743e2aeb4a9a959))
* **headtail:** cascade stage 5.1 from valid laser labels ([2b73d28](https://github.com/UCSD-E4E/fishsense-lite/commit/2b73d2843e44c3df210b8fe6e71f16da2152610a))

## [1.31.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.31.0...fishsense-api-workflow-worker-v1.31.1) (2026-05-03)


### Bug Fixes

* **laser:** instrument + bump timeouts on validate-laser-labels activity ([570d10f](https://github.com/UCSD-E4E/fishsense-lite/commit/570d10f59b090b6d27c3e7a83c1dc6e8fa580c5c))
* **laser:** timeouts + heartbeats on validate activity; .secrets.toml.example for all workers ([65149fa](https://github.com/UCSD-E4E/fishsense-lite/commit/65149fac706acaf20f8e88e26337e40fcdd9892a))

## [1.31.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.30.2...fishsense-api-workflow-worker-v1.31.0) (2026-05-03)


### Features

* **laser:** post-sync line-fit validation of laser labels (observe-only) ([9fa0e13](https://github.com/UCSD-E4E/fishsense-lite/commit/9fa0e130b83d14d8098b3758d8f9ab3f58f2460c))
* **laser:** post-sync line-fit validation of laser labels (observe-only) ([7390903](https://github.com/UCSD-E4E/fishsense-lite/commit/739090389c8a2e85ba99b34c687a98124982f827))

## [1.30.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.30.1...fishsense-api-workflow-worker-v1.30.2) (2026-05-03)


### Bug Fixes

* **api-workflow-worker,backup-worker:** force fresh login per NasClient ([5655bbb](https://github.com/UCSD-E4E/fishsense-lite/commit/5655bbb8c4c8e62fe639e09d45362d9e42ea50e7))
* force fresh DSM login per NasClient (synology-api class-cache bug) ([458e966](https://github.com/UCSD-E4E/fishsense-lite/commit/458e966138f6cb5bbce6e53d580c7d6da4c963a7))

## [1.30.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.30.0...fishsense-api-workflow-worker-v1.30.1) (2026-05-03)


### Bug Fixes

* **api,api-workflow-worker:** exclude NULL-project sentinels from cohort predicate ([5e09f1b](https://github.com/UCSD-E4E/fishsense-lite/commit/5e09f1bee9cb8f1512fc32253d5be58322118e08))
* exclude NULL-project sentinel rows from cohort predicate ([6380460](https://github.com/UCSD-E4E/fishsense-lite/commit/638046003a0ad62fc784bdff42b9588585905868))

## [1.30.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.29.2...fishsense-api-workflow-worker-v1.30.0) (2026-05-03)


### Features

* **api-workflow-worker:** self-bootstrap populate + drop dives once labels seeded ([e03b7a6](https://github.com/UCSD-E4E/fishsense-lite/commit/e03b7a6ffd0730d4417f80b5e5b4b7e57f043d2e))
* **api-workflow-worker:** self-bootstrap populate + drop preprocessed dives ([a3609dd](https://github.com/UCSD-E4E/fishsense-lite/commit/a3609dd135ffce57964a0e0ef58031874f96084a))

## [1.29.2](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.29.1...fishsense-api-workflow-worker-v1.29.2) (2026-05-03)


### Bug Fixes

* **api-workflow-worker:** emit both `image` and `img` keys in populate task data ([5bc163b](https://github.com/UCSD-E4E/fishsense-lite/commit/5bc163b955d96c68ea004d5359d26daaa5ca0d56))
* **api-workflow-worker:** emit both `image` and `img` keys in populate task data ([7a3109e](https://github.com/UCSD-E4E/fishsense-lite/commit/7a3109e7c770794048ab791255566222c0a509f9))

## [1.29.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.29.0...fishsense-api-workflow-worker-v1.29.1) (2026-05-03)


### Bug Fixes

* **api-workflow-worker:** prepend e4e_nas.raw_root_path to share-relative DB paths ([f4d0195](https://github.com/UCSD-E4E/fishsense-lite/commit/f4d0195a7d4af30eeaabad0998c67b5afae4b76c))
* **api-workflow-worker:** prepend e4e_nas.raw_root_path to share-relative DB paths ([a433383](https://github.com/UCSD-E4E/fishsense-lite/commit/a4333833d15fc5b3d8ae16d679c5f1b9af21bb29))

## [1.29.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.28.0...fishsense-api-workflow-worker-v1.29.0) (2026-05-02)


### Features

* **api-workflow-worker:** re-cohort stage 0.1 by labeling state + fix multi-row predicate ([aeed529](https://github.com/UCSD-E4E/fishsense-lite/commit/aeed529ab340168be644561e8c374ee1dadaeeeb))
* **api-workflow-worker:** re-cohort stage 0.1 by labeling state + fix multi-row predicate ([91a81c7](https://github.com/UCSD-E4E/fishsense-lite/commit/91a81c79386b2661470e2aad71585639824975f6))

## [1.28.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.27.0...fishsense-api-workflow-worker-v1.28.0) (2026-05-02)


### Features

* **api-workflow-worker:** chain populate from preprocess parents + dedup child dispatches ([c4b0f3f](https://github.com/UCSD-E4E/fishsense-lite/commit/c4b0f3f49d5d6abd38f491cad3fd63528a280290))
* **api-workflow-worker:** chain populate from preprocess parents + dedup child dispatches ([17fa7c5](https://github.com/UCSD-E4E/fishsense-lite/commit/17fa7c5ecff555976fc2834fd718b5d813e7e84b))

## [1.27.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.26.0...fishsense-api-workflow-worker-v1.27.0) (2026-05-02)


### Features

* **api-worker:** add stage 13 laser-calibration parent workflow ([5f9a44a](https://github.com/UCSD-E4E/fishsense-lite/commit/5f9a44a39ca77a5f1da30de94935fda7bf077afb))
* **api-worker:** add stage 14 measure-fish parent workflow (on-demand) ([01d9364](https://github.com/UCSD-E4E/fishsense-lite/commit/01d936435b379245a6992c2f2edb3981e6fb10b6))
* **api-worker:** wire stage 13 + 14 parents to data-worker ([01c4809](https://github.com/UCSD-E4E/fishsense-lite/commit/01c480986e04385425ac4db4686864bac964b19f))


### Performance Improvements

* **api-worker:** collapse dive cohort selectors to single SDK call ([62b6d13](https://github.com/UCSD-E4E/fishsense-lite/commit/62b6d1358a33a71ba6d15505d7ab05a4880e0b1a))
* **api-worker:** collapse dive cohort selectors to single SDK call ([fab3157](https://github.com/UCSD-E4E/fishsense-lite/commit/fab3157acf840f91b7a5ac7f73664ae178b64f4c))

## [1.26.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.25.0...fishsense-api-workflow-worker-v1.26.0) (2026-05-02)


### Features

* **api-worker:** apply parent/child pattern to stages 2, 5.1, 9 ([a22c5c5](https://github.com/UCSD-E4E/fishsense-lite/commit/a22c5c5988c8d2c2cf9e7fe0f57d3b466ca25500))
* **api-worker:** Phase 3a — stage NAS bytes to file-exchange before dispatch ([42c7b0a](https://github.com/UCSD-E4E/fishsense-lite/commit/42c7b0a510a5701ebfd0eebcf2fb9bbffaea6a07))
* **api-worker:** Phase 3b — archive JPEGs to NAS + raw cleanup ([4079e39](https://github.com/UCSD-E4E/fishsense-lite/commit/4079e398f8045a07714eadb302140ef9803224ec))
* **api-worker:** stage 0.1 parent + cluster-safe schedule ([504b05f](https://github.com/UCSD-E4E/fishsense-lite/commit/504b05f03e489abda424aabf0fdef5043c669346))


### Documentation

* **api-worker:** pull FIFO-ordering note into selector docstrings ([9ecdd58](https://github.com/UCSD-E4E/fishsense-lite/commit/9ecdd5817b8ba2debabacb4a70bbe0b0d8c9db3f))
* CLAUDE.md, deploy/README.md, api-worker README. ([4079e39](https://github.com/UCSD-E4E/fishsense-lite/commit/4079e398f8045a07714eadb302140ef9803224ec))

## [1.25.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.24.0...fishsense-api-workflow-worker-v1.25.0) (2026-05-01)


### Features

* **api-worker:** port stage 4.2 sync_species_labels ([91d17f4](https://github.com/UCSD-E4E/fishsense-lite/commit/91d17f44d7266fae2338fb6e3c8bef364940eb7b))
* **api-worker:** port stage 6.1 update_dive_image_groups ([0ae07a2](https://github.com/UCSD-E4E/fishsense-lite/commit/0ae07a2e375246d84e246447d624bd4ea51ce55b))

## [1.24.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.23.1...fishsense-api-workflow-worker-v1.24.0) (2026-05-01)


### Features

* **api-worker:** port stage 12 sync_slate_label ([7922716](https://github.com/UCSD-E4E/fishsense-lite/commit/792271617a33bfc9c5907151814a02d3f21853ff))

## [1.23.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.23.0...fishsense-api-workflow-worker-v1.23.1) (2026-05-01)


### Bug Fixes

* **api-workflow-worker:** heartbeat during slow LS-task listing ([62a2a91](https://github.com/UCSD-E4E/fishsense-lite/commit/62a2a910940ed35bad94f32ab161161dc886e060))

## [1.23.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.22.1...fishsense-api-workflow-worker-v1.23.0) (2026-05-01)


### Features

* **api-workflow-worker:** add Create + Populate Label Studio project workflows ([ab3fde5](https://github.com/UCSD-E4E/fishsense-lite/commit/ab3fde54144be88cd1c5997052c54ef196074ea9))
* **api-workflow-worker:** wire labeling-config XML for all four LS create activities ([9386057](https://github.com/UCSD-E4E/fishsense-lite/commit/9386057fa4edc144854ce8d3052cc6b354e348a2))
* **deploy,api-workflow-worker:** add Label Studio to local stack + integration tests ([09d4e6b](https://github.com/UCSD-E4E/fishsense-lite/commit/09d4e6b9a5e829b97ddf0f1c368e86ebdd9fb114))


### Bug Fixes

* **api-workflow-worker:** bump per-project sync timeout for backlog runs ([ec97a5b](https://github.com/UCSD-E4E/fishsense-lite/commit/ec97a5b1765c1939eae0292ca92083e2902ff51b))
* **api-workflow-worker:** clean up pylint errors in LS-populate code ([2c1a006](https://github.com/UCSD-E4E/fishsense-lite/commit/2c1a006c96e98e2e15caebe2db0e0ac9995c8fcc))

## [1.22.1](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.22.0...fishsense-api-workflow-worker-v1.22.1) (2026-05-01)


### Bug Fixes

* **api,sdk,api-workflow-worker:** single-call distinct-project-ids endpoint ([cef2d4d](https://github.com/UCSD-E4E/fishsense-lite/commit/cef2d4d1386d3d544680a9f49b6e78c75b596d7d))

## [1.22.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.21.0...fishsense-api-workflow-worker-v1.22.0) (2026-05-01)


### Features

* **api,sdk:** incremental Label Studio sync via LabelStudioSyncCursor ([726a147](https://github.com/UCSD-E4E/fishsense-lite/commit/726a1477e436c8ca0235e60e2354e4154b1dac22))


### Bug Fixes

* **api-workflow-worker:** bound sync concurrency, add heartbeats, extract helper ([40e04fe](https://github.com/UCSD-E4E/fishsense-lite/commit/40e04fe4b4bfeeccb7258f49914f9a6b44914c5a))

## [1.21.0](https://github.com/UCSD-E4E/fishsense-lite/compare/fishsense-api-workflow-worker-v1.20.1...fishsense-api-workflow-worker-v1.21.0) (2026-04-29)


### Features

* **ci:** build-once / promote-tag deploy split + monorepo-aware Dockerfiles ([66d20b9](https://github.com/UCSD-E4E/fishsense-lite/commit/66d20b9b9bc834bc2ebc88461006d71a7dd14faa))
* **ci:** root monorepo workflows (lint, release, docker) ([d58b806](https://github.com/UCSD-E4E/fishsense-lite/commit/d58b806b2ebf40e4826c7c601c8bd8ffd4fa4790))
* **shared:** extract fishsense-shared lib for Dynaconf, logging, TLS, ExceptionGroup helpers ([f896c5f](https://github.com/UCSD-E4E/fishsense-lite/commit/f896c5fc6017edc509e0e0c651da2b2a4c6519e6))


### Documentation

* fill in package-level READMEs across services and libs ([df477db](https://github.com/UCSD-E4E/fishsense-lite/commit/df477dbb4c0956d4aa3864c66a2ffc13a31a9feb))

## v1.20.1 (2026-02-25)

### Bug Fixes

- Handle 404 for dives that don't yet have labels
  ([`54e563d`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/54e563d445b1f32efdb1b6ff60f81a41127759d8))

- Pylint errors
  ([`5de9482`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/5de94820e816295349d7f2cd1476e37dccd646fe))

### Chores

- **deps**: Bump pillow from 12.1.0 to 12.1.1
  ([`8443ca8`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/8443ca831462faad5db9ce4e3fa36f52f1d18bf6))


## v1.20.0 (2026-02-24)

### Chores

- Additional cleanup
  ([`0379eb9`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/0379eb903d945127c5623290e226b3289ab267ca))

- Cleanup repo
  ([`2351f45`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/2351f45431bc966c86de3fe690e9510780d482dd))

### Features

- Handle a user which doesn't exist
  ([`b29f801`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/b29f8011cf015ea622c0e6ea95f91331ea1fa4d1))


## v1.19.1 (2026-02-08)

### Bug Fixes

- Update api sdk to resolve bugs
  ([`4e69c58`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/4e69c585322ec4888e9bbf6980284a25a350132c))


## v1.19.0 (2026-02-08)

### Features

- Add Copilot instructions for repository
  ([`ef2b334`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/ef2b334c4fff597aa4d930b77c1255d0ce6db89d))


## v1.18.7 (2026-02-08)

### Bug Fixes

- Update fishsense api sdk to deal with warnings
  ([`e1ea398`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/e1ea398ec1f3867a06e735b56d9e8016743c1b3e))


## v1.18.6 (2026-02-08)

### Bug Fixes

- Update to latest fisthsense api workflow worker
  ([`7b5c739`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/7b5c7392ed1b6eaa3b9440b4c0f4e9bb4e6ef1c7))


## v1.18.5 (2026-02-08)

### Bug Fixes

- Update the fishsense-api-sdk package again for bug fixes
  ([`48dc803`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/48dc8037ae83f49deec2280444381ca6ae9bfca9))


## v1.18.4 (2026-02-08)

### Bug Fixes

- Try dumping the json instead
  ([`60fe6ee`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/60fe6ee8304a081c2d0a98c8c95701e00188f78a))


## v1.18.3 (2026-02-08)

### Bug Fixes

- Json validation errors for json blobs from label studio
  ([`4a699c1`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/4a699c1a866962b5cae48c56cb3a8281c3788312))


## v1.18.2 (2026-02-08)

### Bug Fixes

- Update fishsense-api-sdk to pull in bug fixes for warnings
  ([`6aa591a`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/6aa591a2c17f485c3b20115fa21579d6a870eb97))


## v1.18.1 (2026-02-08)

### Bug Fixes

- Properly log aysncio task group issues
  ([`7cc7c03`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/7cc7c03c927da4ea3c9fbb8d7dfd9aa3e8a1a59f))


## v1.18.0 (2026-02-08)

### Features

- Introduce logging to get_label_studio_projects_activity
  ([`16dae50`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/16dae5008e9302c3755f06c436fa90da123ace4c))


## v1.17.5 (2026-01-05)

### Bug Fixes

- Update sdk to include logging
  ([`fb6f341`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/fb6f34136721cbea8ea964aecae90d5c823f64cc))


## v1.17.4 (2026-01-05)

### Bug Fixes

- Update fishsense-api-sdk to reduce warnings
  ([`de21a88`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/de21a88afa9982fe858fae5fa94f3f3cb90d5c65))


## v1.17.3 (2025-12-24)

### Bug Fixes

- Context manager correct configuration
  ([`54efbe9`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/54efbe95a67bcc86f314768bc56721fd926866bd))

- Handle exceptions in workflows as well
  ([`b856abb`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/b856abb17121a2cd261595fcd7961af07e116e1c))

- Try again to do error reporting for task group
  ([`4c809ca`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/4c809ca24424abb5e145946dbcf9601c1f4f5f0c))


## v1.17.2 (2025-12-23)

### Bug Fixes

- Order of decorators for activity
  ([`8880ca3`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/8880ca3dfe65a153206f9471b7f2d9e4d266abc6))


## v1.17.1 (2025-12-23)

### Bug Fixes

- Allow workflows to timeout
  ([`aac4068`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/aac4068b103bf98aa2fef634c33c0480c135b749))

- Use the correct timeout defintion for scheduling workflows
  ([`28a42bb`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/28a42bb8c0a994eb62e8bbb76246d164e9278f35))


## v1.17.0 (2025-12-23)

### Features

- Decorator for task group error reporting
  ([`1e6e900`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/1e6e900dac9c3dd25e6b03a81b2568a5de18250a))


## v1.16.1 (2025-12-23)

### Bug Fixes

- Better logging for exceptions which occur in sync headtail and laser labels
  ([`3e75892`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/3e758925e6ee1765ca7e864e42f635845e987a7c))


## v1.16.0 (2025-12-19)

### Features

- Introduce headtail label sync task
  ([`a40ef46`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/a40ef462fe93c5bba646be253d05d306e7095baa))


## v1.15.3 (2025-12-17)

### Bug Fixes

- Correct the order of the dashboard
  ([`33ce302`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/33ce3028160669f3e1c2b411b643151d0cc67186))


## v1.15.2 (2025-12-17)

### Bug Fixes

- Serialization error for label studio project
  ([`0be3107`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/0be31071f3254c1d6f7f319c85d7519cc50758a7))


## v1.15.1 (2025-12-17)

### Bug Fixes

- Add missing workflow to schedule
  ([`be5f280`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/be5f2809b71a7903011aa2adbb056ef022796e1d))


## v1.15.0 (2025-12-17)

### Features

- Introduce workflow to create the config file for dashboard
  ([`df929e0`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/df929e08d6d1a3f7d44b9478c79de306e1ce71e2))


## v1.14.1 (2025-12-03)

### Bug Fixes

- Pass pylint
  ([`5248e2e`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/5248e2e3fe02b915e9554872d70cc6fef389f8b5))

### Chores

- Code cleanliness
  ([`ef1ecb2`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/ef1ecb230dac7ee217bdc9d7cb0566094f8651a4))


## v1.14.0 (2025-12-02)

### Bug Fixes

- Finish adding flake support
  ([`1a1af77`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/1a1af77fc927f38ce6f20e24f13fc5c7c03c1ce4))

- Remove flakes. readonly file system issue with notebooks
  ([`af568da`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/af568dafc40ef617ca0c58e4e323f74d4ab4130d))

### Chores

- Introduce codeowners
  ([`9e389b3`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/9e389b3046ff114060f26eaf35ec00e2375694a9))

### Features

- Call data worker workflow
  ([`cf00025`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/cf0002597fab864b652ece8cd315186d99b78211))

- Ingest dive workflow
  ([`8f4e5b0`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/8f4e5b0a7b7a88c6f397843c86c2e3c4fd6ecbd8))

- Introduce flakes
  ([`6b470c5`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/6b470c55da2b4191d5ba55cf758cd4c2da472cdd))

- Start cleanup in preparation for updating superset automatically
  ([`3d07042`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/3d07042c074eb1f018c260ee1c6979fa7daa56ef))

- Support clustering dive frames
  ([`3c12d55`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/3c12d55d246e65f4f1edce287110c1826fc07ace))

- Sync laser labels updated
  ([`1032591`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/1032591f8427ab89c7761e5400d33614b8df3ecf))

- Sync users during sync laser labels
  ([`53b1ab5`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/53b1ab502aa0bc19a0ce5e443619353c1f01e5d8))


## v1.13.0 (2025-09-15)

### Features

- Remove the scheduled workflows
  ([`ef05013`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/ef050138122d6605bd8369bb1b9a676a581cc8ca))


## v1.12.0 (2025-09-09)

### Features

- Mtls on worker
  ([`948f3e4`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/948f3e4aefd57dd8909ca66174e1f3555edd65ae))


## v1.11.1 (2025-09-06)

### Bug Fixes

- Move json property so that it no longer shadows json on the base
  ([`c0e7b7f`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/c0e7b7fb4c5d8d21c86efec36554c33e3d4d6dbd))


## v1.11.0 (2025-09-06)

### Features

- Add dive slate
  ([`1d1547a`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/1d1547af04341ea77113a5780bc3740f964c3383))


## v1.10.1 (2025-09-04)

### Bug Fixes

- Make sure user exists in laser label before trying to use it
  ([`0409300`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/04093009c581f5e6c44eebd5d957b2d13690a114))


## v1.10.0 (2025-09-04)

### Bug Fixes

- Pylint issues
  ([`78ac102`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/78ac1023dbb3f37933d3cd880db842819e7c9121))

### Chores

- Finished migrating laser labels
  ([`d716b79`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/d716b792d52213017d8c572a9e80b758c73255de))

### Features

- Add completed tag to labels
  ([`e2597a4`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/e2597a4172b67331ad969743795550f224196319))


## v1.9.2 (2025-09-03)

### Bug Fixes

- Merge collect and insert activities so that we don't need to share data.
  ([`2f65b58`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/2f65b581852b975350df0877ba16f57a15a92ac1))

- Pylint errors
  ([`67a8965`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/67a8965ce7f707ce0cd4616b9e6d731b01b7ac4b))


## v1.9.1 (2025-09-03)

### Bug Fixes

- Update workflows to timeout after 20 minutes instead of 10
  ([`06ec086`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/06ec08693a269c78474f24a463f7b4822e99d85e))


## v1.9.0 (2025-09-03)

### Features

- Add missing user label studio id to database
  ([`e733291`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/e733291e6d6f6f5c7f9b9b7af69888c9a8f0410c))

- Touch to force a new minor version
  ([`b394240`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/b39424080b6eee4645ae2830150b0fb1c737c762))


## v1.8.0 (2025-09-03)

### Features

- Add missing user label studio id to database
  ([`ddd9bc3`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/ddd9bc3b0f39def132694d87c732b661a9ce7456))


## v1.7.0 (2025-09-03)

### Features

- Add json and updated date to label studio labels
  ([`158e633`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/158e633a0f4c9ed4f237feac909318218ecac3ca))


## v1.6.0 (2025-09-03)

### Bug Fixes

- Touch to force a new version
  ([`41c2d8d`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/41c2d8d1be02dd0359167495ee804a75ab712512))

### Chores

- Readme update
  ([`9e985ef`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/9e985efa9d99602bd95fcbd62343694f2221752d))

- Update readme again
  ([`b83b677`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/b83b677722697d87018e94a108097d42e745ed03))

### Features

- Add alembic
  ([`49a1d27`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/49a1d271d9cf1695c6748c5b9d27da1ad4959a6c))

- Add alembic for database migrations
  ([`98fe307`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/98fe3077a8120a94c2607a590123be084d4a361e))


## v1.5.8 (2025-09-03)

### Bug Fixes

- Use timezone utc in database
  ([`69c8298`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/69c829863d732f035bfd1df73a54b6cb5e72bf65))


## v1.5.7 (2025-09-03)

### Bug Fixes

- Use timezone aware times for user creation in label studio
  ([`720b87f`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/720b87f284fd6b17e642fa98032a8fc05e6347d8))


## v1.5.6 (2025-09-03)

### Bug Fixes

- Pydantic allow null user ids
  ([`275fd59`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/275fd59b4fb70bbc0eb1783d68308828f19b7c60))


## v1.5.5 (2025-09-03)

### Bug Fixes

- Failed run for collecting user b/c we did not specify a url
  ([`a5c5a3f`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/a5c5a3f8da014623ad30d8b90e867332ac05ceac))


## v1.5.4 (2025-09-03)

### Bug Fixes

- Please pylint
  ([`4354ca3`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/4354ca36bd7e9752f053c7d64ffc3a42950b0055))

- Pydantic doesn't like the label studio api type
  ([`3be187d`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/3be187d313646aeac72a4f475b9e11ba7cccd87d))


## v1.5.3 (2025-09-03)

### Bug Fixes

- Pydantic errors with label studio api
  ([`20f0507`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/20f0507d24cfc646fe16013a8aec115c18dc9e6c))


## v1.5.2 (2025-09-03)

### Bug Fixes

- Move label studio import into class.
  ([`79c158b`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/79c158b6cb61e619ffd47c40559454ba35859ea6))

- Please pylint
  ([`233d5b0`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/233d5b092ea94aa43a7edf4da1bf60373c38e22e))


## v1.5.1 (2025-09-03)

### Bug Fixes

- Pylint errors
  ([`461e2f8`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/461e2f88e4d2fe518ea90808686320987bd9d326))


## v1.5.0 (2025-09-02)

### Features

- Add tqdm
  ([`9e8e52b`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/9e8e52bf9e2096205a5fc0e0986529e5f5b29caa))


## v1.4.1 (2025-09-02)

### Bug Fixes

- Add executor for sync activities
  ([`11ec9b9`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/11ec9b9bbdb3b7d3ff487f89ede2074d1b25424c))


## v1.4.0 (2025-09-02)

### Bug Fixes

- Don't initialize the database in a loop
  ([`a8e734d`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/a8e734d5e77ad4c7a2124f6d0666442857c427bb))

- Ensure that we set users for labels
  ([`7bd6465`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/7bd6465eeac3be451637de61cac86ea7a5f62a72))

- Other copilot related errors
  ([`451a20e`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/451a20e6eb9c8ca70cdc9c61a7a7b5cb45fa53a3))

- Run uv sync when starting container
  ([`439faf4`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/439faf4ef84497191129ef63661896b8d569959b))

### Chores

- Appease pylint
  ([`41cc142`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/41cc142c3d2a7dce7babcbc99966907e9db47ad8))

- Cleanup unused code.
  ([`26b1ac1`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/26b1ac174a67a0fc7051f27ea85f7c3a765158f2))

- Please pylint
  ([`79a1857`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/79a185776d94662e89677149e4b271c54cf6aade))

### Features

- Add head tail label
  ([`987a1ae`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/987a1ae2d90202a7cad831ffac500abc11bc4e1e))

- Add users to database
  ([`941958a`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/941958a9eee1870dbaa1f7ea1555e78d7957dbce))

- Can initialize database
  ([`8dc99ba`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/8dc99ba6a7e38b8803c8a2bcf1c2186a6581a664))

- Initialize database
  ([`12df08d`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/12df08df99ab1c74b1c7d2a2208ee95d8cd4d0ad))

- Introduce postgres to the dev environment
  ([`00966ec`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/00966ec6f28ae2a6121c55f492149fcb6ee6f2de))

- Start of dev container
  ([`730c780`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/730c780bf5669e463cc5f5a3c18cb90cd2e071a1))

- Support cameras, dives, and images fully
  ([`9e5bc64`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/9e5bc64ac43324d9456c2669cac9a98df20f763a))

- Support head tail labels in new database
  ([`6492143`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/6492143ddd49f75db2889d6f88a091c62806a674))

- Sync users from label studio
  ([`a5cee52`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/a5cee522413b8e1e1b6d22dfd10dc420bb52c69e))

- Update activity to use new orm database layer
  ([`83b94df`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/83b94df2175d4d61e3a9bc68f644e60cb9ce6c9b))


## v1.3.1 (2025-08-22)

### Bug Fixes

- Label studio percentages are in whole numbers.
  ([`349f5d2`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/349f5d28cf52cbcae80de4af5c1a4749914aec5a))


## v1.3.0 (2025-08-22)

### Bug Fixes

- Pylint
  ([`841613c`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/841613ccde2c34272746b3d34118b88b49696fb3))

### Chores

- Fix pylint
  ([`ab068ea`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/ab068ea656c777e8c3ea9357c2001a850da727e3))

### Features

- Collect head tail labels from label studio
  ([`51d6037`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/51d6037df33075ff0bb8c2085574d47a0e464241))

- Insert headtail labels into postgres
  ([`c1d52f3`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/c1d52f33cdec5010c6bb2ed1e20b0a7c2e68e6da))


## v1.2.4 (2025-08-21)

### Bug Fixes

- Sql file name
  ([`3ec1a34`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/3ec1a34609e6a7c485278b0bf69e1c6e2a4fc102))


## v1.2.3 (2025-08-21)

### Bug Fixes

- Pylint error
  ([`fcfad9b`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/fcfad9bafd15e3ebe1cfe9ecd2d83512df93c945))

- Use the full path if running in docker
  ([`b688b07`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/b688b0709e636f2a72c72dc3ef77e52d2558cad9))


## v1.2.2 (2025-08-21)

### Bug Fixes

- Sql path issue in docker container
  ([`016fba7`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/016fba7cf9e5534f3a5d81dbd9df1ab0abdc7a11))


## v1.2.1 (2025-08-21)

### Bug Fixes

- Add missing sql scripts
  ([`9180389`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/918038931c2deaa4950ff9d3921047643a63e730))


## v1.2.0 (2025-08-21)

### Features

- Implement label studio laser activity
  ([`92f57f9`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/92f57f9d7f54e02a8080f01d4b648d28b84132b8))

- Introduce logging
  ([`f527c18`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/f527c18b7a6f0dd3911eee57e5075eecc229b1cf))

- Write to postgres
  ([`0b40ed0`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/0b40ed0219e73a313d10332c2ff211efd4d54073))


## v1.1.0 (2025-08-21)

### Features

- Add validators to settings
  ([`1ba107b`](https://github.com/UCSD-E4E/fishsense-api-workflow-worker/commit/1ba107bc9b5cf6f998fc14ee59239f850aabbf80))


## v1.0.0 (2025-08-21)

- Initial Release
