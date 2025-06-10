# CHANGELOG

<!-- version list -->

## v1.0.0 (2025-06-10)

### Bug Fixes

- Add bump to dockerfile
  ([`e3346dc`](https://github.com/UCSD-E4E/fishsense-lite/commit/e3346dc9a17edc4a125bcf612e3c194399097096))

- Add calculate laser coord to pipeline
  ([`48f67b2`](https://github.com/UCSD-E4E/fishsense-lite/commit/48f67b26d1d23dff5ea1a6a859447d2689c574cb))

- Add missing setter for psql_connection_string for detect slate for dives
  ([`ca5bcc0`](https://github.com/UCSD-E4E/fishsense-lite/commit/ca5bcc098868f684b6d599070822c1267bb5d696))

- Apt-get instead of apt
  ([`460d358`](https://github.com/UCSD-E4E/fishsense-lite/commit/460d3580f96fe637cbda5db4987c2747a594ecc5))

- Auto versioning
  ([`cde9347`](https://github.com/UCSD-E4E/fishsense-lite/commit/cde934713e286b8fc9e71613ea4e6646fd981e87))

- Avoid debug conflicts
  ([`418c123`](https://github.com/UCSD-E4E/fishsense-lite/commit/418c1232bbcbc2dce4a007f4a94b57193573c122))

- Bug causing jpgs to be white when preprocessing.
  ([`c0dde50`](https://github.com/UCSD-E4E/fishsense-lite/commit/c0dde50f9ea7aafc4be88e8e62df640561461572))

- Bug that prevents creating json file if jpg already exits.
  ([`ed709da`](https://github.com/UCSD-E4E/fishsense-lite/commit/ed709dad354af32c016588a46c5b396b25cf34e9))

- Can't use -p for prefix in label studio
  ([`d9f699e`](https://github.com/UCSD-E4E/fishsense-lite/commit/d9f699e1ad25d2bb896705b27cca1eacc38af449))

- Cast to ubyte before displaying the laser
  ([`394cef3`](https://github.com/UCSD-E4E/fishsense-lite/commit/394cef347c944dcb994a5bc5fcbf8d8f4dc3f9be))

- Check for none and exit early
  ([`9b70ded`](https://github.com/UCSD-E4E/fishsense-lite/commit/9b70ded2e3122c6732490ba571a68a2407b775a1))

- Check if laser_coords is None
  ([`dddc3f9`](https://github.com/UCSD-E4E/fishsense-lite/commit/dddc3f9b7604e00689eeffc8ce9a229343606855))

- Check rawpy for error
  ([`12cbb03`](https://github.com/UCSD-E4E/fishsense-lite/commit/12cbb039206fbd84837a0f9a27f99286bcdf084b))

- Convert string keys to path
  ([`a09b05d`](https://github.com/UCSD-E4E/fishsense-lite/commit/a09b05dd51ffa89f45fe37abf2c01ccd0ef3c479))

- Correct device
  ([`efcaa9d`](https://github.com/UCSD-E4E/fishsense-lite/commit/efcaa9dacd461a64a00197d1c0b33b2bdcdcc770))

- Correct typo in json format
  ([`cc1e929`](https://github.com/UCSD-E4E/fishsense-lite/commit/cc1e929cf9af069ff274841e4a7ec6301aac1439))

- Correctly return status for calculating points of interest
  ([`e22785c`](https://github.com/UCSD-E4E/fishsense-lite/commit/e22785ce082d4e6ff11e742fcb443dfee13db79f))

- Default port number if not specified
  ([`1db035e`](https://github.com/UCSD-E4E/fishsense-lite/commit/1db035e186740a7d6f361e8f4cf3e43b9ceec10f))

- Delete tools cache
  ([`a639ded`](https://github.com/UCSD-E4E/fishsense-lite/commit/a639dedf737b702a7cbbd2d5650e816048168a5b))

- Devcontainer works correctly
  ([`9655568`](https://github.com/UCSD-E4E/fishsense-lite/commit/96555683c3dde76942e194827b80b0ff569af4b0))

- Docker container build
  ([`7655318`](https://github.com/UCSD-E4E/fishsense-lite/commit/7655318e4c1df643e36b1747590659ef9bfb335a))

- Docker container build
  ([`35df60d`](https://github.com/UCSD-E4E/fishsense-lite/commit/35df60de6bf85ab80781d8bbe9d3ada84a4be48b))

- Dockerfile pull from right container
  ([`15af32d`](https://github.com/UCSD-E4E/fishsense-lite/commit/15af32d4a0b8c9489d13708d2914f53e3e2358f0))

- Don't attempt to calibrate if we do not have enough 3d laser position
  ([`890e24a`](https://github.com/UCSD-E4E/fishsense-lite/commit/890e24a2506e69cccaa7f33dcf768c49a31843c1))

- Don't build docker container in GH actions
  ([`7c4bdfe`](https://github.com/UCSD-E4E/fishsense-lite/commit/7c4bdfe985d4d3a8a5f6a171e4ac756487e68cc2))

- Don't calibrate if we have less than 4 valid points
  ([`f62243a`](https://github.com/UCSD-E4E/fishsense-lite/commit/f62243aeb6c0edd936cb0bb3b4c905f912787b52))

- Don't encode the prefix.
  ([`7d1a8b4`](https://github.com/UCSD-E4E/fishsense-lite/commit/7d1a8b488b1e009d79537ebbb8d214015ab81aa5))

- Don't pylint yet
  ([`17e1ab8`](https://github.com/UCSD-E4E/fishsense-lite/commit/17e1ab8a74b026bbf3d8474379b6b9f156c1a2e2))

- Don't reference lens calibration in many of the method calls for calculate laser coord
  ([`76c4355`](https://github.com/UCSD-E4E/fishsense-lite/commit/76c435541bcd19ffaf77377db93f486684ac9577))

- Duped keyword argument
  ([`b36cacf`](https://github.com/UCSD-E4E/fishsense-lite/commit/b36cacfc57cbd6da29f49e8069adf1cc0055d0ab))

- Ensure the correct bit depth is passed aroundg
  ([`143f07c`](https://github.com/UCSD-E4E/fishsense-lite/commit/143f07c278448e59286cf9c5dfcef9bbac9b813a))

- Get the cuda group not the groupname group
  ([`3628a35`](https://github.com/UCSD-E4E/fishsense-lite/commit/3628a3524a7edda85df89bce9b82c074e73d5f5e))

- Handle img and laser image coords being null in display laser
  ([`ec5be45`](https://github.com/UCSD-E4E/fishsense-lite/commit/ec5be45191daef4a600d7b559a5b23c0182b3203))

- Handle not finding laser
  ([`f89336d`](https://github.com/UCSD-E4E/fishsense-lite/commit/f89336dadfc6fe726436cb210eb28ad4facd6a9d))

- Handle not finding the laser
  ([`8010912`](https://github.com/UCSD-E4E/fishsense-lite/commit/8010912b3240049e9aceefface2c6e2ecdf16259))

- If lens or laser calibration does not exist, return none
  ([`6ff4679`](https://github.com/UCSD-E4E/fishsense-lite/commit/6ff46797736e5976111d9f808ac878041dd0991a))

- Img not img-1 for target
  ([`d033dca`](https://github.com/UCSD-E4E/fishsense-lite/commit/d033dcafdfbd7d0189a8950bc0ad666a816ddc5f))

- Install libopencv-dev
  ([`e8e3798`](https://github.com/UCSD-E4E/fishsense-lite/commit/e8e3798c2f27b46f825b4ab5694e8c791b7cd804))

- Label studio expects 8 bit masks, not boolean
  ([`6f8badd`](https://github.com/UCSD-E4E/fishsense-lite/commit/6f8badd4671d69315ac21ec797178f32deecb852))

- Label studio looks for a list of objects
  ([`686bc16`](https://github.com/UCSD-E4E/fishsense-lite/commit/686bc162eac8b718e119924a2c6e76ed3e3493bd))

- Make sure to export the type we expect for dives
  ([`d3ab510`](https://github.com/UCSD-E4E/fishsense-lite/commit/d3ab5108ceefb037d607ef6ee86557cef45886c0))

- Make the debug path
  ([`1ad4793`](https://github.com/UCSD-E4E/fishsense-lite/commit/1ad4793016101e89a3b7bc3fb42d1c8bd982a3d0))

- Missing argument
  ([`ac25d01`](https://github.com/UCSD-E4E/fishsense-lite/commit/ac25d01909cb9ad9aebdf0f5942a8ba69fe23037))

- Move bump to before install
  ([`bc114ee`](https://github.com/UCSD-E4E/fishsense-lite/commit/bc114eebe850fa6508d2ece5c64dd21ab4b22042))

- Need ray default not ray core for kube cluster
  ([`e1e91b0`](https://github.com/UCSD-E4E/fishsense-lite/commit/e1e91b06a23c74ee92656befabc589f44098d0f9))

- Only store a single json task per file
  ([`cb54a98`](https://github.com/UCSD-E4E/fishsense-lite/commit/cb54a98c7c7f47eecd80bbda742918d92f147837))

- Pass in pdf object not path
  ([`65f9052`](https://github.com/UCSD-E4E/fishsense-lite/commit/65f90521fe47938b35138ce1a3dc97df649db46e))

- Pass in the module name for finding the version
  ([`9cd6ba6`](https://github.com/UCSD-E4E/fishsense-lite/commit/9cd6ba62d0749092e741758cd8e06109e071d39c))

- Prefix should not be required
  ([`d3a360a`](https://github.com/UCSD-E4E/fishsense-lite/commit/d3a360a9a34c0d245287c9668f1eb1c65348efd9))

- Print more information during process
  ([`48f3b7b`](https://github.com/UCSD-E4E/fishsense-lite/commit/48f3b7bffeb61d3c8f4730259a5f29871bb098b7))

- Processing issues
  ([`c68e11d`](https://github.com/UCSD-E4E/fishsense-lite/commit/c68e11db71f6c67ea8c4f5b0a4f1981745dadbae))

- Proper tuple for points of interest
  ([`be751c7`](https://github.com/UCSD-E4E/fishsense-lite/commit/be751c7db1caa6250928c4548ed00cff424caf40))

- Pyproject toml
  ([`227a9a2`](https://github.com/UCSD-E4E/fishsense-lite/commit/227a9a208f76f16b196491e54e936b7119603119))

- Reference error
  ([`a0cbe7b`](https://github.com/UCSD-E4E/fishsense-lite/commit/a0cbe7bedacc8b3c13a2b02ee9384106e41a8d93))

- Reference the static file
  ([`d0f4ce1`](https://github.com/UCSD-E4E/fishsense-lite/commit/d0f4ce1b9206a8eef83e901845008a6853e26a46))

- Register field calibrate laser job
  ([`0add893`](https://github.com/UCSD-E4E/fishsense-lite/commit/0add893b14403d8c68d53daf456a430e115ec89e))

- Return a proper job count
  ([`4655123`](https://github.com/UCSD-E4E/fishsense-lite/commit/465512360d4c94e43cd10e0c429ad12f5a0664fb))

- Revert to using a dict instead of a list
  ([`a98bf1d`](https://github.com/UCSD-E4E/fishsense-lite/commit/a98bf1db5e50c9122ca786821a58cfcf9487ed3c))

- Shm-size
  ([`455b56b`](https://github.com/UCSD-E4E/fishsense-lite/commit/455b56ba170db6b44cf002139d657b3a455df640))

- Specify git repo
  ([`dce9c5d`](https://github.com/UCSD-E4E/fishsense-lite/commit/dce9c5d0bd703724b6a707dd6ae73c63287d45c3))

- Support early exit for preprocess
  ([`0368a71`](https://github.com/UCSD-E4E/fishsense-lite/commit/0368a71b426aaa26563156cd5598ff77de8c3658))

- Support proper globs in preprocess
  ([`96ab752`](https://github.com/UCSD-E4E/fishsense-lite/commit/96ab7522031fcc60649196317774abd3ffaf9ab9))

- Support vulkan in docker container
  ([`b222ed9`](https://github.com/UCSD-E4E/fishsense-lite/commit/b222ed90f95600c1cad6a9c4525e3507ba96071c))

- Switch raw processor module
  ([`e904dbc`](https://github.com/UCSD-E4E/fishsense-lite/commit/e904dbc669cc2284520ceef42cfb5b1c4b88ca71))

- Switch to not building runtime on github actions
  ([`81b700d`](https://github.com/UCSD-E4E/fishsense-lite/commit/81b700d9adc7043dc21d4347b245bc3e1b3caee1))

- Too many values to unpack
  ([`cf5d7c1`](https://github.com/UCSD-E4E/fishsense-lite/commit/cf5d7c1746126e9c07cc806d3f134e72f1a0edd4))

- Try to install fishsense.rs
  ([`1cc8016`](https://github.com/UCSD-E4E/fishsense-lite/commit/1cc8016bc8cde41234fe5c6f9af244ed04208ff0))

- Typo
  ([`ae8741c`](https://github.com/UCSD-E4E/fishsense-lite/commit/ae8741c1f099985cf50275a622b132b551fb52e3))

- Typos
  ([`8f2fb9f`](https://github.com/UCSD-E4E/fishsense-lite/commit/8f2fb9f0f908bf7fb0a25e2e77bf012be316105e))

- Up the vram requirement for nn_laser label studio
  ([`8c6414f`](https://github.com/UCSD-E4E/fishsense-lite/commit/8c6414f030c14f57c510468b1674d229493bb731))

- Up the vram requirement for nn_laser label studio
  ([`708a2ce`](https://github.com/UCSD-E4E/fishsense-lite/commit/708a2ce305ebf55ab31b8a7573c595786aa1bd2b))

- Update raw processing and laser detection
  ([`de881b4`](https://github.com/UCSD-E4E/fishsense-lite/commit/de881b421fbd21e4b153423bdd7104fb0e3fb3fb))

- Use git commit id for bump
  ([`08ace50`](https://github.com/UCSD-E4E/fishsense-lite/commit/08ace50395eaff002977f22b0fc8fc40cc5a26ce))

- Use red for laser instead of blue
  ([`87d55b4`](https://github.com/UCSD-E4E/fishsense-lite/commit/87d55b40c9a0a8e2af5ca3ac703e67e624b837ba))

- Use string as key for dictionary
  ([`677ff1e`](https://github.com/UCSD-E4E/fishsense-lite/commit/677ff1eee6830e8befa010771dc7d8a337debf71))

- Use sudo to install libopencv
  ([`817957e`](https://github.com/UCSD-E4E/fishsense-lite/commit/817957e19bb3e05a6c840b357ba11e7734ce1543))

- Use the old raw processor correctly
  ([`89a7ba9`](https://github.com/UCSD-E4E/fishsense-lite/commit/89a7ba9edce391a0e5b1737703fe5f837df03d24))

- Use the old raw processor in preprocess.
  ([`17b0b89`](https://github.com/UCSD-E4E/fishsense-lite/commit/17b0b894366411601cef904ce8a5f21aea8c3aca))

- Use the right orientation of the pdf for the debug images
  ([`650d1ba`](https://github.com/UCSD-E4E/fishsense-lite/commit/650d1ba92577a4ae1628930e0639119ef389e38c))

- Wrong attribute name.
  ([`010807a`](https://github.com/UCSD-E4E/fishsense-lite/commit/010807a33a0f83f4ab12aad6e6def15f5a7261b3))

### Chores

- Create dependabot.yml
  ([`c28d6ca`](https://github.com/UCSD-E4E/fishsense-lite/commit/c28d6cad31a93abc7019c72f74e7acf380ad19bf))

- Poetry lock
  ([`5888d71`](https://github.com/UCSD-E4E/fishsense-lite/commit/5888d716753c2e0cba4e71d264e2aefbad24db80))

- Poetry lock update
  ([`535127e`](https://github.com/UCSD-E4E/fishsense-lite/commit/535127e60ba60d6ca77b6fdecd3b67b19d66612d))

- Poetry update
  ([`59ac833`](https://github.com/UCSD-E4E/fishsense-lite/commit/59ac8339d9d7536459bd3771844dcc3055e2a532))

- Poetry update
  ([`4304997`](https://github.com/UCSD-E4E/fishsense-lite/commit/430499708c7e6f2997f89b16f9e10936860f1bda))

- Poetry update
  ([`776fa65`](https://github.com/UCSD-E4E/fishsense-lite/commit/776fa653714082f4698037dafd8386037e0eb17e))

- Poetry update
  ([`6bb601b`](https://github.com/UCSD-E4E/fishsense-lite/commit/6bb601bdbb5ed84df3f8367da5b4e828875c82f8))

- Poetry update
  ([`9fa120e`](https://github.com/UCSD-E4E/fishsense-lite/commit/9fa120e557880300ac7425e8b0f7b4cb84e5f479))

- Poetry update
  ([`3c81647`](https://github.com/UCSD-E4E/fishsense-lite/commit/3c816472fa125233287d1b1a9a994db1b85323b4))

- Poetry updatE
  ([`96a2a48`](https://github.com/UCSD-E4E/fishsense-lite/commit/96a2a488d3d1f08406a05ca2899688077c8b37ad))

- Poetry updatE
  ([`0d71cf0`](https://github.com/UCSD-E4E/fishsense-lite/commit/0d71cf0c31729b2e6307fe1f7c4b248145c73436))

- Poetry update
  ([`b7af997`](https://github.com/UCSD-E4E/fishsense-lite/commit/b7af9973a9757db5d9aaec447b3bdbf6bfbac2ed))

- Poetry.lock
  ([`1c9bbb9`](https://github.com/UCSD-E4E/fishsense-lite/commit/1c9bbb99913730d1ad9889d06fb5bd42f031931f))

- Poetry.lock update
  ([`d9215d6`](https://github.com/UCSD-E4E/fishsense-lite/commit/d9215d62d5412bf1ec3e84811516cf694d7a68b2))

- Poetry.lock update
  ([`b01222e`](https://github.com/UCSD-E4E/fishsense-lite/commit/b01222e4db13664b6ad482d4e6eff72478b7d8ae))

- Pylint fixes
  ([`f2219a3`](https://github.com/UCSD-E4E/fishsense-lite/commit/f2219a346fc224d4559141f52b06a76388e835a1))

- Rename
  ([`adc1668`](https://github.com/UCSD-E4E/fishsense-lite/commit/adc16688a56fb72dd03bab887570bde611ed6dd6))

- Update
  ([`3f6e5df`](https://github.com/UCSD-E4E/fishsense-lite/commit/3f6e5df40fcfd17eebf858ad16de310bace9a894))

- Update fishsense-common
  ([`ee4b72d`](https://github.com/UCSD-E4E/fishsense-lite/commit/ee4b72d9792d15b46714604cf1b23eb781401fd9))

- Update fishsense-common
  ([`c58beb7`](https://github.com/UCSD-E4E/fishsense-lite/commit/c58beb7d25a2bb0beb8b3ee71d3454d1e01d74b9))

- Update packages
  ([`694d35b`](https://github.com/UCSD-E4E/fishsense-lite/commit/694d35b2f1c9663b6c34a8d04e176b697803c4e0))

- Update poetry lock
  ([`011445b`](https://github.com/UCSD-E4E/fishsense-lite/commit/011445b49a2672cd321f6a46e6e67fce86bf4a1f))

- Update poetry lock
  ([`d3d49e7`](https://github.com/UCSD-E4E/fishsense-lite/commit/d3d49e7b4e0456c8166485faf12b59c6a1a9e6a1))

- Update poetry lock
  ([`89c16cc`](https://github.com/UCSD-E4E/fishsense-lite/commit/89c16cca20d95b3b15eb6de943a9e861955886f9))

- Update poetry lock
  ([`667e559`](https://github.com/UCSD-E4E/fishsense-lite/commit/667e559885a96ccc11bbeb79fe5648cff6945032))

- Update poetry lock
  ([`09f3ed6`](https://github.com/UCSD-E4E/fishsense-lite/commit/09f3ed608161d091b3c087704e07ed3407a4fcac))

- Update poetry lock
  ([`8fa4054`](https://github.com/UCSD-E4E/fishsense-lite/commit/8fa405473b9ff600815e90480d3568c668bf6c83))

- Update poetry.lock
  ([`2f3de7c`](https://github.com/UCSD-E4E/fishsense-lite/commit/2f3de7c3843481d35ef57c16f34bbfa8fa7a3cfa))

- Update poetry.lock
  ([`5723eaf`](https://github.com/UCSD-E4E/fishsense-lite/commit/5723eaf8accd4e427dbd0043516e80413087445e))

- Update poetry.lock
  ([`0596044`](https://github.com/UCSD-E4E/fishsense-lite/commit/05960441bc8414594c1dd0c563e79480a6680356))

- Update poetry.lock
  ([`9549aeb`](https://github.com/UCSD-E4E/fishsense-lite/commit/9549aebee0e1ea46004ad17778e3db1b78b0972b))

- Update poetry.lock
  ([`5c2dbdf`](https://github.com/UCSD-E4E/fishsense-lite/commit/5c2dbdf711e9b97f3cc69064e661a18a648ba36c))

- Update poetry.lock
  ([`8f874e5`](https://github.com/UCSD-E4E/fishsense-lite/commit/8f874e599711283090b8dc427234b689d9ba3407))

- Update poetry.lock
  ([`4218434`](https://github.com/UCSD-E4E/fishsense-lite/commit/42184344b43eddfaf37c70baa1926673c5408c2d))

- Update poetry.lock
  ([`ccd8713`](https://github.com/UCSD-E4E/fishsense-lite/commit/ccd8713f35d08d072be1177faa4d1a943e5418e7))

- Update poetry.lock
  ([`4a898c0`](https://github.com/UCSD-E4E/fishsense-lite/commit/4a898c0cd7485e43f3076453c87258c2299ac952))

- Update poetry.lock
  ([`d3cfca8`](https://github.com/UCSD-E4E/fishsense-lite/commit/d3cfca82e5ed44d287c27572ae8d860071fc7372))

- Update poetry.lock
  ([`17463d1`](https://github.com/UCSD-E4E/fishsense-lite/commit/17463d19fd2353048b23ddcf002603f600239e4d))

- Update poetry.lock
  ([`d283a8c`](https://github.com/UCSD-E4E/fishsense-lite/commit/d283a8c7c4988e0d79b410f0e66be34d75517d60))

- Update poetry.lock
  ([`65a97f6`](https://github.com/UCSD-E4E/fishsense-lite/commit/65a97f68f9a6ad621ba9e051d6e362ec15be962d))

- Update poetry.lock
  ([`190a015`](https://github.com/UCSD-E4E/fishsense-lite/commit/190a01536d2a9b771243af1f9a88d6c9ace4b754))

- Update poetry.lock
  ([`d3f2754`](https://github.com/UCSD-E4E/fishsense-lite/commit/d3f275470cbf8881b4a954b2e27136fb73f480c6))

- Update pyfishsensedev
  ([`ed7f374`](https://github.com/UCSD-E4E/fishsense-lite/commit/ed7f37499060e67db96fbe6cc8dc5c366515f6ea))

- Update pyfishsensedev
  ([`10a34ca`](https://github.com/UCSD-E4E/fishsense-lite/commit/10a34ca6ef9ecec33c6fecf0ee5828132915e76b))

- Update pyproject.toml and poetry.lock
  ([`ba6d280`](https://github.com/UCSD-E4E/fishsense-lite/commit/ba6d28099b87170361c704e8052d09516efb650f))

- Upgrade to python 3.12 and cleanup dev container
  ([`3660dda`](https://github.com/UCSD-E4E/fishsense-lite/commit/3660dda3d55712a48a2baee48495939ac0f82ac9))

- **deps**: Bump requests from 2.32.3 to 2.32.4
  ([`4457668`](https://github.com/UCSD-E4E/fishsense-lite/commit/4457668eddd1d16da89c7df24b77ecbe8caa2362))

- **deps-dev**: Bump tornado from 6.4.1 to 6.4.2
  ([`5e5895c`](https://github.com/UCSD-E4E/fishsense-lite/commit/5e5895c66d4720ce9228559e6c6bcdb8487c5c34))

### Features

- Add gitignore
  ([`540e308`](https://github.com/UCSD-E4E/fishsense-lite/commit/540e3087998542dcd3fcb44c4e91c67c938c5730))

- Allow rotating the pdf
  ([`c5324be`](https://github.com/UCSD-E4E/fishsense-lite/commit/c5324be0e9787d6445c35bdd9bf04f70c5b4c00b))

- Allow stopping of execution and recovery
  ([`aacf914`](https://github.com/UCSD-E4E/fishsense-lite/commit/aacf914ea8f1a0c2f96b2b0d95efc33e336ef746))

- Better distribute the vram across the gpus
  ([`781bc2f`](https://github.com/UCSD-E4E/fishsense-lite/commit/781bc2f7f12b1471f5fc3b426a1ef6b8baf94967))

- Better raw processing and seathru
  ([`e4b9c89`](https://github.com/UCSD-E4E/fishsense-lite/commit/e4b9c89059e4f5b03f17a824969d28c26f09a61f))

- Bump again
  ([`db0e3de`](https://github.com/UCSD-E4E/fishsense-lite/commit/db0e3de0f45b7edea17a28684f3d72ca27d6f704))

- Create a json file per photo
  ([`725058f`](https://github.com/UCSD-E4E/fishsense-lite/commit/725058f795a5ea8c866078c91e399d7680a28d53))

- Cudnn in dockerfile
  ([`1d30917`](https://github.com/UCSD-E4E/fishsense-lite/commit/1d309171cb89995570df82e83de85412e365e545))

- Define preprocess as tasks
  ([`2872c7e`](https://github.com/UCSD-E4E/fishsense-lite/commit/2872c7e79ca58b7518bb32a63046085581a45c42))

- Demo with smb for preprocess
  ([`5aed8e1`](https://github.com/UCSD-E4E/fishsense-lite/commit/5aed8e15c2fcafb4570202e9f1b4b754b2f851c6))

- Error measurements for both lens and laser calibration
  ([`3ad5507`](https://github.com/UCSD-E4E/fishsense-lite/commit/3ad55072bc2c2137688f771537c8cc46b0ec9fa9))

- Fallback to not using seathru when seathru fails
  ([`7fffd82`](https://github.com/UCSD-E4E/fishsense-lite/commit/7fffd82a5ae89b66852876a5c8dac0c239ee070a))

- Field calibrate laser
  ([`bf8cbc8`](https://github.com/UCSD-E4E/fishsense-lite/commit/bf8cbc8a9a9705e821a716a0a1890e31e199f129))

- Full processing pipeline
  ([`86fa251`](https://github.com/UCSD-E4E/fishsense-lite/commit/86fa25101e7bc08cbb82960ee96003eddd8ee7cc))

- Generate fishial masks
  ([`531ab0c`](https://github.com/UCSD-E4E/fishsense-lite/commit/531ab0c6ed348417c8b67ff96800a219179e8a2c))

- Handle camera calibration in fsl cli
  ([`3a074dd`](https://github.com/UCSD-E4E/fishsense-lite/commit/3a074dd423b589f23ea3f605ea33ebef49de73c2))

- Introduce preprocess with laser label
  ([`de98ea5`](https://github.com/UCSD-E4E/fishsense-lite/commit/de98ea59fa1a7335081b9d8d21fcb6c7d230bfcc))

- Keep slate counts instead of raw names
  ([`80d0fa4`](https://github.com/UCSD-E4E/fishsense-lite/commit/80d0fa4a00921ecc2f429b26c2e917caaa475520))

- Label studio generator
  ([`c1c447d`](https://github.com/UCSD-E4E/fishsense-lite/commit/c1c447d3defffb64964042c7c159c7ddb1aa1c39))

- Make the laser label yellow. check for pdb better
  ([`9391649`](https://github.com/UCSD-E4E/fishsense-lite/commit/9391649f17736550d6e8c2026b1b42085009f9c7))

- More none checking
  ([`508699a`](https://github.com/UCSD-E4E/fishsense-lite/commit/508699a4f2a4f00b3ebb0731c1613915ca11222c))

- Move some of the torch imports inside methods to work towards improving performance
  ([`3229336`](https://github.com/UCSD-E4E/fishsense-lite/commit/3229336d819105bb2246e21ce5b46d1da9dec2a9))

- None checking
  ([`e6f3467`](https://github.com/UCSD-E4E/fishsense-lite/commit/e6f34679bd5bba6345e7ab4b4fae07a305528f71))

- Output mask as part of execution for fishial
  ([`ec1de42`](https://github.com/UCSD-E4E/fishsense-lite/commit/ec1de42fe4a871606052ae34ac68728c6a81a7fc))

- Preprocess allows using fsspec
  ([`67eb354`](https://github.com/UCSD-E4E/fishsense-lite/commit/67eb354816b11889e42ed6b62ac8f347c05e5e93))

- Print each item added to db
  ([`515e866`](https://github.com/UCSD-E4E/fishsense-lite/commit/515e866292d20fe1fa94db12cabc8402780f28b8))

- Print the up vector on the slate debug
  ([`34fba74`](https://github.com/UCSD-E4E/fishsense-lite/commit/34fba747a01feabf29707f4ca34971c406ed7da2))

- Pylint
  ([`4865bc7`](https://github.com/UCSD-E4E/fishsense-lite/commit/4865bc7118cecc9d00949d5a1e6a01f974a7e964))

- Readd debug
  ([`0eecf8f`](https://github.com/UCSD-E4E/fishsense-lite/commit/0eecf8f6c7c03d3485477316159d688f0906ad64))

- Round the up vector in the display
  ([`d0d706d`](https://github.com/UCSD-E4E/fishsense-lite/commit/d0d706dbdee6fc4215d73a3e756d978a2b752df2))

- Scale based off of how much vram we use
  ([`23177f5`](https://github.com/UCSD-E4E/fishsense-lite/commit/23177f5c11c188f8637937e97ce839c620f5b348))

- Semantic release
  ([`14fff4a`](https://github.com/UCSD-E4E/fishsense-lite/commit/14fff4a0903e003a6f4d8f231feda6500e1dd768))

- Specify psql password as part of the docker run
  ([`723b737`](https://github.com/UCSD-E4E/fishsense-lite/commit/723b737b854ffa73c85291d52b358368b0ce9432))

- Support adding prefix to the output json files
  ([`cc8d820`](https://github.com/UCSD-E4E/fishsense-lite/commit/cc8d820cc848f800c249595ddb665174c32b5091))

- Support detecting slate for dives
  ([`b27fc7d`](https://github.com/UCSD-E4E/fishsense-lite/commit/b27fc7d3269227dedb7300e5d1a031b85241fbbc))

- Support preprocessing with laser using psql
  ([`0e00096`](https://github.com/UCSD-E4E/fishsense-lite/commit/0e0009602c9d1e944f6f6e1630f2dbef49ed5651))

- Support trying multiple rotations of the dive slate
  ([`4f7ef78`](https://github.com/UCSD-E4E/fishsense-lite/commit/4f7ef78c83a2fe6d30434375bb41a588668debf1))

- Switch to using jobs instead of cli. first job has been created for preprocess
  ([`4c2586f`](https://github.com/UCSD-E4E/fishsense-lite/commit/4c2586f1731d2fefda354270a5f3937cb0829ffa))

- Turn on bump
  ([`f1f9fe9`](https://github.com/UCSD-E4E/fishsense-lite/commit/f1f9fe961b1e0f2f0b60cb54439464b5299d4e51))

- Update docker build script to better ensure we have access to gpus
  ([`c545a0d`](https://github.com/UCSD-E4E/fishsense-lite/commit/c545a0df7fe0be957b839bd41086b0aa64976d67))

- Update to use upstream fishsense docker container instead of one built here
  ([`377ac38`](https://github.com/UCSD-E4E/fishsense-lite/commit/377ac3814c33e0bcca0e834c01079215c2f0b66e))

- Use devcontainer features of vscode instead of setup script.
  ([`d0b5d89`](https://github.com/UCSD-E4E/fishsense-lite/commit/d0b5d89594e88deb96af378a96049141dd071273))

- Write checkerboard image to debug folder
  ([`eca95ee`](https://github.com/UCSD-E4E/fishsense-lite/commit/eca95eeebf90c40cabf252aff37698a3629c1bda))


## v0.1.0 (2024-11-04)

- Initial Release
