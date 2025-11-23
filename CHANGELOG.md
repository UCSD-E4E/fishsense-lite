# CHANGELOG

<!-- version list -->

## v1.19.1 (2025-11-23)

### Bug Fixes

- Points json should be a list of tuple
  ([`470468a`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/470468adcd68fdfd3ff89223350cba312d588850))

- Support datetime to json
  ([`786aebf`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/786aebf949299141ed01f24483b866e789021b7b))


## v1.19.0 (2025-11-23)

### Features

- Introduce put method for dive slate client
  ([`cce75b8`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/cce75b84c907e01b8dfee99304a4658bd281dd93))


## v1.18.0 (2025-11-23)

### Features

- Add dpi property to dive slate
  ([`13941f3`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/13941f3b5ad196e309f713a285c7d6a3d10e1faa))


## v1.17.0 (2025-11-23)

### Features

- Introduce methods to get dive slates from api
  ([`4f253fb`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/4f253fb7f5a592defbe4712bba8e960cfb8907d8))


## v1.16.0 (2025-11-23)

### Features

- Allow selecting headtail labels with dive id
  ([`235e410`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/235e41049e0f59fa11764a44741007100328ba31))

- Support getting canonical dives
  ([`083ff27`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/083ff2773d69108f098e9a3711b37f1e39916180))


## v1.15.0 (2025-11-23)

### Features

- Support querying images by checksum
  ([`c265991`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/c265991f323b22d6897b944d928ffecb0d509922))


## v1.14.0 (2025-11-23)

### Features

- Add put headtail label
  ([`92340a7`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/92340a7eaa4514540ae0a52480c435b86aff04f7))


## v1.13.0 (2025-11-23)

### Features

- Introduce headtail label
  ([`48b0849`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/48b0849a4aa53c430641978f69992376f74ce078))


## v1.12.0 (2025-11-22)

### Features

- Add put laser label
  ([`9184d86`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/9184d867316a77dadc0a5ccfe326615b2e077ad4))


## v1.11.0 (2025-11-22)

### Bug Fixes

- Client object itself should be an async context manager
  ([`1c9cd4c`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/1c9cd4ca22188c546af7a6a4cba37b5a0f1b4168))

- Label_client now properly supports using context manager
  ([`cbcf299`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/cbcf299843ddd160a3ef401ec0afdad237629b00))

- Make get dive work correctly
  ([`e64d89f`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/e64d89fe46655c5903dd6f3d4e9c0bb9c8754abe))

- Make sure all the clients are functional again
  ([`ab2f60c`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/ab2f60c8e784b203ee12321a04e27fd548fab8a5))

### Features

- Introduce a max number of concurrent connections
  ([`ddad7dd`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/ddad7dddf11bb73c66bba597071ac705444de6a3))

- Use async with for client
  ([`f763d30`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/f763d30fc654123fe1b9274b3302c921e255f23c))


## v1.10.0 (2025-11-22)

### Features

- Implement autoretry for get_laser_labels
  ([`f7bac23`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/f7bac231f753aaf8450fb28440018552178fae87))


## v1.9.0 (2025-11-22)

### Features

- Introduce laser_labels api
  ([`2eb4bf3`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/2eb4bf3c698c472f819302be1a44bb5dc44f203b))


## v1.8.1 (2025-11-18)

### Bug Fixes

- Better match label studio
  ([`402d4b8`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/402d4b8d74c58119c0cf14f280e348ceac0671b8))


## v1.8.0 (2025-11-18)

### Features

- Add additional species label properties
  ([`29264e2`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/29264e29cba469753520a4cf75cf9d78661a96ca))


## v1.7.1 (2025-11-18)

### Bug Fixes

- Use mode json for species label
  ([`572bdcf`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/572bdcf75817c0e155eb20e2dcca1b16535df72a))


## v1.7.0 (2025-11-18)

### Features

- Support getting user from label studio id
  ([`86d6fd2`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/86d6fd22d9b592bee583db820d776d7ab76a7b10))


## v1.6.2 (2025-11-18)

### Bug Fixes

- Use dump mode json instead of producing a json string
  ([`675e386`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/675e3867628c5c5b36456ad497c0a5c5b46ac804))


## v1.6.1 (2025-11-18)

### Bug Fixes

- Use model_dump_json so that datetime is handled
  ([`c90d223`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/c90d22383fbadc7cf00ad6eaba03e406ddbdb1a7))


## v1.6.0 (2025-11-18)

### Features

- Introduce post method for user
  ([`f11d4b6`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/f11d4b6d2efae8a2b5cf99dca423dbe0e232dc5d))


## v1.5.2 (2025-11-17)

### Bug Fixes

- Url path for putting the user to
  ([`0ec7a3b`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/0ec7a3b8ccfe4ccc0dde86941947be9005c17dfd))


## v1.5.1 (2025-11-17)

### Bug Fixes

- Switch to aware date time
  ([`3c22abe`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/3c22abe9888984c07752e5afb4be452dee6ec52f))


## v1.5.0 (2025-11-17)

### Features

- Allow getting users by email
  ([`bc4c3df`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/bc4c3dfaa61268351c4ad9ec72fda98e72dc5fc5))


## v1.4.1 (2025-11-17)

### Bug Fixes

- 405 error
  ([`5d2fac1`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/5d2fac1585e70dee6824e177e24a04ac85050bc3))


## v1.4.0 (2025-11-17)

### Features

- Add user client
  ([`45c9265`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/45c926530427fd98f966a2b4c690ac3f974e65d2))


## v1.3.0 (2025-11-17)

### Features

- Introduce the ability to get species labels by dive
  ([`ffc4267`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/ffc4267f39bb8a822695f96c14b6c61deb33370d))

- Update model for dive frame cluster to include datasource and updated at
  ([`3bfac48`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/3bfac4884a71bebfa832d7d256ae470894f956fe))


## v1.2.1 (2025-11-16)

### Bug Fixes

- Handle cases where there is nothing returned from the api
  ([`31df34a`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/31df34a194b972edb0a9155a22a7eb50d6466d91))


## v1.2.0 (2025-11-10)

### Features

- Add label_studio_project_id
  ([`77c9969`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/77c99696512d303a41adda4006877d6e78e700b1))


## v1.1.1 (2025-10-30)

### Bug Fixes

- Remove flakes. readonly file system issue with notebooks
  ([`cc65e41`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/cc65e419030010a392cba5865e54c11dc47237b1))


## v1.1.0 (2025-10-30)

### Chores

- Add flake.lock and codeowners
  ([`d95c93f`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/d95c93f447e830dbffa5d7eb387559b657e9938d))

- Update uv lock
  ([`8edc474`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/8edc47412de7c4af2f7f29b555dfdf5578ce9dd9))

### Features

- Introduce flakes
  ([`ef20130`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/ef201302e75ba5d9e9d20f471129dbd8328ea411))


## v1.0.1 (2025-10-22)

### Bug Fixes

- Use an older version of numpy to not break opencv
  ([`4249663`](https://github.com/UCSD-E4E/fishsense-api-sdk/commit/4249663a59338a46e0321fe8f4e00741e1d72c45))


## v1.0.0 (2025-10-22)

- Initial Release
