# CHANGELOG

<!-- version list -->

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
