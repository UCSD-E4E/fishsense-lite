# CHANGELOG

<!-- version list -->

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
