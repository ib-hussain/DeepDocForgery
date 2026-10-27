# Supplied-file inventory

The original archive contained 231 files. Every original file was read for hashing and classification; text JSON/JSONL records were parsed, Python source was parsed into definitions, and training failures were traced through the relevant source. Binary caches/assets are not claimed as executable source review. No dataset images or `.pt` model checkpoint were supplied.

| Original path | Bytes | Lines | SHA-256 | Audit treatment |
|---|---:|---:|---|---|
| `.gitattributes` | 2546 | — | `d6669ffb578ab7a004de878392119dfc9e5a408d579dede33833af1ac4e2a7dd` | Documentation/configuration/metadata inventory; no repair mutation |
| `.gitignore` | 906 | — | `1100c1217d0622458d59d6c93c4ffaa5cef7f504a8457ddb12a6103ff99dc1dc` | Documentation/configuration/metadata inventory; no repair mutation |
| `AGENTS.md` | 16038 | 309 | `e8b0f0f052026eb48aa454c014ad4d2aed0b2adb183ccc82e0b77352f29d7dce` | Documentation/configuration/metadata inventory; no repair mutation |
| `CHANGELOG.md` | 8727 | 155 | `bfc508f016b4f70938b8efdeb83f5e7bb102358e244d1c617c5ba0244ba0ca95` | Documentation/configuration/metadata inventory; no repair mutation |
| `LICENSE` | 35149 | — | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | Documentation/configuration/metadata inventory; no repair mutation |
| `MEMORY.md` | 4836 | 109 | `db6f63319e413dc1abd1229243489d8b9d3196f7bb4bdb3e965932dc0e602f45` | Documentation/configuration/metadata inventory; no repair mutation |
| `MIGRATION.md` | 3437 | 69 | `9772b6d4d753d1afc9e39cc99de66f80b1d90ec122935467c323c36121d8c5c6` | Documentation/configuration/metadata inventory; no repair mutation |
| `README.md` | 14281 | 288 | `067b96e214a218e334243bf6171ea871ba60526f97ab512190c212fa7aff19e0` | Documentation/configuration/metadata inventory; no repair mutation |
| `SKILL.md` | 9514 | 220 | `95bcee367737f0c5817885964613d7cccb3aa515e347c78cdf6dd47f663c1734` | Documentation/configuration/metadata inventory; no repair mutation |
| `deepdocforgery/README.md` | 705 | 14 | `6c23a8867e384ce2d331d616ee780b23f9a37760fda0a750625272a939f3e6da` | Documentation/configuration/metadata inventory; no repair mutation |
| `deepdocforgery/__init__.py` | 230 | 6 | `6efc4400d792f4f5bfcc3bd09da33e86c2aab5e83a3ee0fcf95bf17fd9e6971d` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/__main__.py` | 1815 | 54 | `e89c615521f70ee9419a486dc0b72f76aec8c11806f6e3c5005aa30e968f6709` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/data.py` | 34669 | 801 | `c68be5f9855b373980809223068aeb91919355137a1b76820e4ad5dd9abf86b9` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/decoder.py` | 12887 | 314 | `182a2cc76ef82faebe9d0e10520dbab9415b23e38748d26ec774a521cd09ac4b` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/degradation.py` | 17801 | 453 | `a8d797d8743bfb39a4686e1c919c3de8f61e3c9a5197ee27fb73a9c162654c03` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/doctor.py` | 18854 | 514 | `fb94a0e96332c52579a860fe97ae6aa51fe53aae7a31d2d0be5d0fca9d49acb5` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/evaluate.py` | 14290 | 402 | `06c12e345aac13169371ecd565a7f5a0e43e03c794c69b4f6cec9b0b5347c1d3` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/frequency.py` | 22804 | 548 | `50f0b3f462d2e6027366f88df49a781588619abbcbb3466370da3f728c03af35` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/fusion.py` | 15847 | 396 | `ba80d9dc9bdc3fe56f5954dddb6704021bc61a8797bcbaa3da985002d55ff3c8` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/hpo.py` | 18637 | 507 | `7286bf8e177baa91042b93a5e1f97f1f691f9ef73dab6bc25efd82096a372140` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/infer.py` | 17829 | 483 | `1388fad140d3540df7691c3c7814547fc3360ccb76770a4c9ceb643e8fe53026` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/io.py` | 16673 | 441 | `dc797bcab7aa2a0731d8947bd4497080c18219f877a15cf0052fdd163363d6aa` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/losses.py` | 10947 | 266 | `f011bcd1d31ec0dc936c23dab7db946ab35deef7f023e2634c2e1eac4d5e8008` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/metrics.py` | 15374 | 352 | `94d9e748ea8576690bd8b5d255c4e09e621f01be1a5c7e600314e92cbb34c02a` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/model.py` | 2771 | 86 | `b233359f1419f3fe6a5a7d351b396bf863394b0cd70360388735e1e263c39fa6` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/objectives.py` | 5167 | 133 | `46db7737038d81d24a0ec6d632eb537c580c92b76ea0c6fe18ff80967caf412a` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/postprocess.py` | 4007 | 108 | `b629334790925d7c3ff3396d27fd188c0fa988fc387b120a185d079e081d823d` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/prepare.py` | 46154 | 1267 | `d54db79404d7a0712c9e1837d9453c54020b0f113831c97b335680fba4851230` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/runtime.py` | 20990 | 494 | `1c53cdb751329cc4b81597bbc080ff9d73ced4627495fe2a638614817381b55c` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/smoke.py` | 7878 | 199 | `cf8a53048ef608f83efe1b2081675e7477eb183d031e497a22287b56666658d4` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/spatial.py` | 31360 | 811 | `852c47858bbd1c6c99147282e5a0ea3b30f9aa37b2aa11b25961d468ed3b14f5` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/state.py` | 12248 | 360 | `249da9dfff9427e66c0334f1acfe1b49d152dea9bf68378ceb4789df2bfb22ea` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/status.py` | 6661 | 178 | `357763430b97fa4af51449be437ee2c07afe7f8745ecc0a71405fa23ffdd9505` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/telemetry.py` | 18409 | 502 | `59ada955d12383feea55f0f8cb5d78952e5587361ab835e7e4e5e958bbc13931` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `deepdocforgery/train.py` | 39287 | 948 | `13a447738e0635ff3b8c59637c181976f8af4d6a58927d727955cdd67e15974c` | Source; syntax/definitions inspected, training interfaces and relevant call paths reviewed |
| `output/README.md` | 700 | 15 | `8ff0b4e6be8b0bd412dac016e2c9186f506916744717aec17f313bbbbd6e8cb0` | Pipeline metadata/log; parsed as supporting evidence |
| `output/embeddings/README.md` | 113 | 4 | `7c8ca58423954b412dbf37480ee1cd48c305e09e339ddc614b1fb8afef65d91b` | Pipeline metadata/log; parsed as supporting evidence |
| `output/hpo/.hpo.lock` | 156 | — | `58bea5350a8baf744ec170d4e7db44c772dfac70d23d906271a13a3ce07aabb5` | HPO configuration/result metadata; separate from main training |
| `output/hpo/README.md` | 664 | 13 | `11f2e0fa8903a3a52d452ad9e5d3ed97cbec1f27e4ad93793a9f57bc366e4328` | HPO configuration/result metadata; separate from main training |
| `output/hpo/study-c50fc0c4111f/state.json` | 26709 | 837 | `4782c970ab67876169226c1403ecbdbf5a4a568a1e3ded0d70b56f97ca785a4e` | HPO configuration/result metadata; separate from main training |
| `output/inference/README.md` | 350 | 7 | `130a3e829d034365eb31712ce4e8fbb3f47539a225d597ea30e2c162f4fc4730` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/README.md` | 883 | 17 | `b5f84b5c866084256322ae983296af8c4f3352b751eabac34a1c206eda73dd90` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/doctor/20260909T055304Z-18614-825740.jsonl` | 5750 | 10 | `0993a27bb188bd1ced886e381d3c0a8970c800d311f892d74745ea401cc34fc0` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/doctor/20260909T055304Z-18614-825740.log` | 785 | 10 | `5e04b2b0a8e64be5b2acc0dd31aed9fc410461f63c5d3a286cb6f48eb4d99e01` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/doctor/latest.json` | 1673 | 56 | `53e8e1c18d2ca7b2b31255529221124e59c3596aa1e53fc7794e1b389f44db8a` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/doctor-cpu.json` | 1215 | 52 | `60074679d0e25ab1639ad524fe76ff2345c81afce27ff10be9eb8c636b263ad5` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/doctor-cuda.json` | 5032 | 186 | `5b1c478831861812f20aba3553230851fa021b07925511a9e6b14b22eda272a9` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/hpo/20260916T032605Z-23240-3a340f.jsonl` | 34406 | 27 | `8138682410c7397570928273935ba8464150ed0f9283db6c1c329602887e48c3` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/hpo/20260916T032605Z-23240-3a340f.log` | 3601 | 27 | `f7b431c8303e16daf20f79297743da63ba919809fba32cb23882320098f60e4b` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/hpo/latest.json` | 1762 | 55 | `fb1882c7240760182b22a76e0b735a36e77f39fc53cbb9f5e24d765d0f85ff20` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/prepare/20260909T061525Z-31968-474514.jsonl` | 109646 | 194 | `6e2d5badc464b9e5df85d8fb265b45642f9ddd7507b806c62b4e88f02ddc22e2` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/prepare/20260909T061525Z-31968-474514.log` | 18658 | 194 | `5bb161e7f22dcce193e44056a9ae6e4034a51bdca51cabef98aeb1ee26b0659e` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/prepare/latest.json` | 1020 | 29 | `467cea549328e21c11526950766952cd0593f247ff96c0c8527e275fc65d02d4` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/scan.log` | 399 | 19 | `76d10aa271ef2186da4798bbe0baf0f47eb81b3a956c09bab22a0ae12a805f70` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/setup/README.md` | 228 | 5 | `a97df7012a56fe72dc8167fe4aa43a19c52d4e1f785716734a4657695670d794` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/setup/cuda-20260907T081550Z-1390857.log` | 40287 | 419 | `0bcf4573927f24b01bb8ed52afd8693562b0ffd00d65db09d809f8bd2fe9d151` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/setup/cuda-20260908T102859Z-12290.log` | 2309 | 27 | `615ec7d49a068e34d73a4705b77bc6cb23a6af80cd26851e3cdeaf15660b4ffc` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/setup/cuda-20260908T102913Z-12387.log` | 35829 | 403 | `c112fcfd1dac47d3eea3373a8c78a6339919825d1d1696197f9d37ad9fad2861` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/smoke-cpu.json` | 979 | 25 | `ae1ceb5fec89cc261272ea46ca0d0752de847d86dd59a94e5d7a2ae9fb52aa26` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/status/20260909T033844Z-8357-09ed3a.jsonl` | 2517 | 4 | `301d2452afcde5a549803d2ac986ae39775954d29d928fb692145a623a171ed8` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/status/20260909T033844Z-8357-09ed3a.log` | 311 | 4 | `431eb18df2ea98bf3c8129b5709a52158203c748018d8456789fa39ee1b24355` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/status/20260909T055605Z-19186-2477bc.jsonl` | 2525 | 4 | `dce92e1526a33fd32ad2c4952be06848d5fe7ba321d9e7784b3fe5d51b63fc62` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/status/20260909T055605Z-19186-2477bc.log` | 312 | 4 | `32542ca23e449df73b77b4709441762e64f4aed4509ac5b0d9936af3871784c4` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/status/latest.json` | 858 | 28 | `c5a6e5ab4c9801d27a4e8fb7f512a9fb82daebb0f2c8209dc668a0cd5e861ce4` | Pipeline metadata/log; parsed as supporting evidence |
| `output/logs/train/20260915T095243Z-22157-667139.jsonl` | 4359 | 9 | `6dc290cefee8854ce4fa44145cadda8f96b07ce1838d8dbc46a9f1adf5d9bb99` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260915T095243Z-22157-667139.log` | 811 | 9 | `1cbbc14c1ad15c50e0aa9fbfb26e6e4c2f243bf6af29f2792ee7dd708b38580c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260915T095513Z-25074-0708ca.jsonl` | 147657 | 125 | `433f6640de52fcd481b84bdae2cdf39b9f9bd9bffc2df95d1889fc19be7e088e` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260915T095513Z-25074-0708ca.log` | 21173 | 125 | `d2af4f3510e14af4a3a2488132b13b5480230794aeb906303ada49367278177d` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T030925Z-7018-ff84fc.jsonl` | 6945 | 6 | `f5f9de41aa44e1efe705a48525e5045fb435db9e0e405ee5d4e05e5f035ac1ea` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T030925Z-7018-ff84fc.log` | 546 | 6 | `5c9dcfb3928f30dfe64b32117cd7bfb8219e614314b38e51ea9ead9390f8b7e7` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031018Z-8126-15ae50.jsonl` | 8529 | 6 | `65c39ae046f83f16d99c9af3e69230ba8a44abc3deaaf385636741add0de4681` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031018Z-8126-15ae50.log` | 1249 | 6 | `331de86fb2a2e140983db34fd4d55e6df1a8f456a552ecbcbe4f9775ac20b939` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031034Z-8338-53bb3f.jsonl` | 7312 | 6 | `bf93e028cab7ec4dfb96ffd9f0d6dec11b4c37eb1303c563b6fc533e1da736e2` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031034Z-8338-53bb3f.log` | 548 | 6 | `dbc68f7bb365e23a0dac9dac33703688dab347198c3eba05618c227091470d91` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031123Z-8693-d47b4d.jsonl` | 4455 | 9 | `56d6e714f15b6d5d694d3ab839ccfc650c2b147b00c213fecf80e5fe6f0626b4` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031123Z-8693-d47b4d.log` | 806 | 9 | `db2c7f5b6d1f20f2ed9395f66d64c7c7faaeb69ea516d7e35f6a069342c81d17` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031342Z-10598-f13b2d.jsonl` | 12279 | 11 | `97c31e14827a562796890f4908a832febe5f2ddf216a472bd7b59c77fc22bc42` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031342Z-10598-f13b2d.log` | 1603 | 11 | `1f9f4cf95fd08f931160fc5c978f908d2a67fd5c4841924b42f545dd14768d6d` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031600Z-12618-3ec72a.jsonl` | 12274 | 11 | `bff755366840370e6bbb5c345517a34de91547ffa47ab20576b2334f6727e63a` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031600Z-12618-3ec72a.log` | 1603 | 11 | `c54c9836db5ad7161f5bb38900385e672789c601023f4097c63da45051bed6e9` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031749Z-14461-d30bf5.jsonl` | 12273 | 11 | `0289b6cea01a8365f90ead9fb8e2180c6ca0ccb26ce36dc83c0e7b04dc9362df` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031749Z-14461-d30bf5.log` | 1603 | 11 | `db29f346dfa2648783223865c187db7c46f876a3a9f33240916a1d7cc1cbe3e5` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031946Z-16231-7dc571.jsonl` | 12279 | 11 | `047ae9617a9bd66102c2a9f790a17b68026234dd9285d6d6c3a2f33dcaf4dc45` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T031946Z-16231-7dc571.log` | 1603 | 11 | `8669a630d70b56c1263ffcea31806ccbb9c89fc5c55defc45e406411f886f32f` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032127Z-17816-7a0c62.jsonl` | 12278 | 11 | `76c336684828225e51d203a93ec7a24388b00c127794778b715f759a712e69c9` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032127Z-17816-7a0c62.log` | 1603 | 11 | `3368f527ec84ed8907d5d69b7a4cb75689e7b12472e58e6f140ddb7707d40f80` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032308Z-19696-382a44.jsonl` | 4473 | 9 | `d1cec89e452d334ddef10edbd6424fb457b1011bfc8718e297fb4e9c546b1f7f` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032308Z-19696-382a44.log` | 807 | 9 | `f601c6b328e8c78812214723670a492e48613050e26db3aa009030516f94a925` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032410Z-21167-6f9274.jsonl` | 4473 | 9 | `c40095074827853bc06dbc93e2e51324cedafa14a73b08af37254b72de935192` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032410Z-21167-6f9274.log` | 807 | 9 | `1876d0b841abc75dc527d7f4f6cb00e93304c6b200363179a0cbe69e69517769` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032546Z-22795-5526b6.jsonl` | 6836 | 6 | `85badd170aad2c20a3b306e9fdb3e5329bf9cbadb88c02c4cde6a9f20db2dcfc` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032546Z-22795-5526b6.log` | 549 | 6 | `5d13bbcccf064049a3b3bd6f628385c72b4ee178fc7dbff50c831aa04c9a86d3` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032607Z-23331-c31221.jsonl` | 54322 | 55 | `0c42d90d8c692a8c6eaba83eb7d7d2f49ae040bac34e7c69371ea800e24d79c9` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T032607Z-23331-c31221.log` | 6982 | 55 | `837642046d086a8b48db54c260253e05f24469f65e81cb5d5185f3dfad8be76a` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T040849Z-56172-579088.jsonl` | 54344 | 55 | `ea1667804e89cc70b4c5e1634ee503bed879a7b0d9c15a0ad507b9dbfb76e660` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T040849Z-56172-579088.log` | 6941 | 55 | `f5b5fa67283ea65692611f8ec6644c04e715e70e01ee606b7c72f8f288e5572a` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T045252Z-93992-cb8c71.jsonl` | 54382 | 55 | `c1b174d5f85d075e35c8144a04e26a67fdcb93cebf4e96fc3428d5513d1926d5` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T045252Z-93992-cb8c71.log` | 6942 | 55 | `dc9e189c2fe90cf6c769608bc20d95515aae15278465b61a8bb4323cfa1346e9` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T053553Z-131344-8854ca.jsonl` | 54815 | 56 | `7c992268641fc041f59591aa789e8d90b658bd0e7323d7a5d4140fdc058c6fac` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T053553Z-131344-8854ca.log` | 7052 | 56 | `461ddecdc74b60233d15adae5433f55e74af6c44df738f4b1caa29d1468010c8` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T061930Z-168894-1ce376.jsonl` | 54874 | 56 | `ca4bfd7f1d595dc2e805178de9175830a0e087a2b2ec05af1adbc65b9e8432dc` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T061930Z-168894-1ce376.log` | 7020 | 56 | `c5ee542e79ad5da20be53fe200ebca3650b70fb2d5aafad0c1a24c72ca26382f` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T070245Z-206344-b8aa9d.jsonl` | 54798 | 56 | `be604e988314d866b2b31937696105727fc4aa40f474a969885ae740a42ba4a3` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T070245Z-206344-b8aa9d.log` | 7020 | 56 | `66f166a53eef45a98bedaeec7fbfc554d158451a2297738bbf8e1d94741baf0b` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T074602Z-243746-064aec.jsonl` | 54708 | 56 | `b99bce182f4f4f340bbae87e348ff8cc10b5c8bed3efd0f012f83aca54b5bf70` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T074602Z-243746-064aec.log` | 7081 | 56 | `ae37953bc309f406e44dc57d31e478529c27233c75b46e4931b2d82144591f94` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T082921Z-281285-7e6ed9.jsonl` | 54063 | 54 | `bb7e447d9699d7115eb71ec38900efc05a4df4fcfdf8e1e91b59ad1855a8e7d6` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T082921Z-281285-7e6ed9.log` | 6873 | 54 | `6c18f4f981b15728cde9b2c8f9875d3f3d52cbb4ee0080cf2fce4f0c1c7ae983` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T091245Z-318662-bff9aa.jsonl` | 54835 | 56 | `461fe5f0fc0a64441561ea77b8b57f949c8c751aaefe5d09aa2f9810d228f4a6` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T091245Z-318662-bff9aa.log` | 7060 | 56 | `3df54fe7a93c2cd850e14ba417550174e5f4c00829525bdab413f6f6c56b5894` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T095543Z-356563-ca0211.jsonl` | 54031 | 54 | `2deb1c66ad71733d51fdec632700fb376004bab1212c599c9daa5dfab6183241` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T095543Z-356563-ca0211.log` | 6872 | 54 | `1e89a75ea4cc36ec58f960ce497561bc4664e7b5f44b4847226bd92eccd940fe` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T103842Z-393964-7a668e.jsonl` | 54385 | 55 | `7cece5d3eedc4cb2f94c711538d0257e596173b42719eb1781c151ac6a944b78` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T103842Z-393964-7a668e.log` | 6977 | 55 | `3c9a39b95b55be662a9cbfac5b1a9f27762cf387d84a0ee30c9eb6c057d58787` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T112149Z-431166-2fdb59.jsonl` | 47821 | 50 | `189ffa114e1e174a560cc824717953ba2a06a9a87fb1a6bd64c4bd087df56168` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260916T112149Z-431166-2fdb59.log` | 6263 | 50 | `793a38ce68b5380e377422c51c0e6a90f693bf17bc928533f7b7581e54f26532` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260917T050224Z-10619-c3b649.jsonl` | 8180 | 6 | `ab4300f6221932b6ef8645684dd13852a791734a2507340088db16b70470613e` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260917T050224Z-10619-c3b649.log` | 1250 | 6 | `2090e24def35a89fef382dbf1b64dd4ef517834039f8efcc941dd86b63db89fe` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260917T050720Z-14771-346ca6.jsonl` | 1510920 | 1220 | `577090cba6a50fe42e1a1733b6763521e47fc6fc61632f6070c8ef0420c5a17c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20260917T050720Z-14771-346ca6.log` | 209899 | 1220 | `806ab6e74a39e4d1e136029058992097eaabb89e5c1bc41cdbebd5bdcfe1cafb` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T032454Z-9557-1d0d60.jsonl` | 7757 | 10 | `f70fb83ff9b3b2f57c19f94ed65a9b27605e3b7e63dd90140c6589b71f142efe` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T032454Z-9557-1d0d60.log` | 1057 | 10 | `f2b129d0a6effaf537e01de267795c2658d4dc93f9931dcaebcc97dd55db77c6` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T033115Z-12940-395e72.jsonl` | 7982 | 10 | `8f0c6c2bd8fe42413dfcb2fb4f7bdb019612a88b0b4236f6a2b17f4fc2bc648c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T033115Z-12940-395e72.log` | 1068 | 10 | `bbce19f869c164b363416f682c7be7c07d65e057101eedd6cc29baf250b2a8e2` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T034529Z-17657-5522bb.jsonl` | 7616 | 10 | `02ff308f21cb3e0630e9fd1ca9b5f32af18a54869a5f871b0059710d9c5ac435` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T034529Z-17657-5522bb.log` | 971 | 10 | `ed4fddcf054e145ce8991620792eb2a4705835e17583bc1ce85a00715075154c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T035649Z-23712-5e0589.jsonl` | 14867 | 12 | `5447ec313a9efcdb08911897152c6fc2b06d3c23a10fd62ad49cbfc181fd7ea7` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T035649Z-23712-5e0589.log` | 3134 | 40 | `5158427c03446702bbc2adc356640580359bd72ed1e0fa40ab066aec5df854b7` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T045426Z-63068-a71579.jsonl` | 46341 | 48 | `392ce353cf8c874cb8caab321daefc649dbe6e7a66c5808ffe43f0e5b3333d95` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261005T045426Z-63068-a71579.log` | 6750 | 48 | `2f62b4f6b4b70bf8cfb99e99c0e1f21a68d13de50fbd49122328239d2ca1786e` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261006T062414Z-12300-bf42f2.jsonl` | 8499 | 12 | `689f71e47d7f145c712ffaa8a4826b9fa7a075f4c477a84de3a660ec6bde9f02` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261006T062414Z-12300-bf42f2.log` | 1118 | 12 | `c89d31d18254a909031aa9e345d1465af81e6192b9fcd1158bbcac1040531365` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261006T062841Z-16324-24b3df.jsonl` | 35892 | 37 | `bd4db27ca11c4701d044bb2b85e7915fbe92b6147300314e3a813055569fe171` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261006T062841Z-16324-24b3df.log` | 5313 | 37 | `db5a24d419318c3803d3a86a8dbec1cc28c587779d155389ce426fc7cea0bfd4` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261007T033305Z-1736-5fbe45.jsonl` | 54244 | 55 | `1d0c9171dcae9c9a55dc59d353cffa74190e184497c527dd4a1d5870bbdbc6d8` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261007T033305Z-1736-5fbe45.log` | 7999 | 55 | `bb40262911651f60493537eb20f544f0bad132356544b8ac892b4843681adca3` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261008T050914Z-7837-729684.jsonl` | 44545 | 46 | `fc506e28a8b78733719bef53726d4410a9a26bd004f712c3cc0ca3ce4e8c2054` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261008T050914Z-7837-729684.log` | 6541 | 46 | `5bc1057bad3475788921081a43250a1a5e7795f6822e5494a636e58dac30ab5e` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261012T050205Z-9913-c6732b.jsonl` | 45466 | 46 | `34e694463822b0d07dde44922389c29d6a0ca5a018994ba1234ee319d86aeba6` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261012T050205Z-9913-c6732b.log` | 6601 | 46 | `217d9e0f63128fc41c5ea1ca2fd7799357e1c95c7fd80efae4ba5412a20f8c4e` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261013T050534Z-1762-4ba9a5.jsonl` | 45439 | 46 | `1d40d44596dc5bd4df99f4f95701a6e8908b9842abe1bf174498a72b22ef60e3` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261013T050534Z-1762-4ba9a5.log` | 6597 | 46 | `d2e682be1c25fcf2e5ae246f4a80762b08c8847700bb0f0392dee01387e7df48` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261014T050113Z-9888-853c22.jsonl` | 34311 | 45 | `9be0ada3e8a8bdcb39d899b3ad7eb472b1cf9b5195f493a2a27acf9798af9f51` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261014T050113Z-9888-853c22.log` | 5867 | 45 | `b589b9b4dfc1c034bbb870f9a0811749e8ede0f9d96a570a698f5eaf73c25eec` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261015T045902Z-9087-6ad411.jsonl` | 37005 | 38 | `f4e25749817cbbb943761f413b7129dcdd11fa95ecb6860a950c347c61ddb989` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261015T045902Z-9087-6ad411.log` | 5480 | 38 | `7742c1f1d84a12752b0b00c4d6bcf5e769a610ce2665e0a4a1bc65e52c46fe48` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261016T045710Z-9028-bece4d.jsonl` | 29336 | 32 | `f763ad92f3771579254f91c25033c79fada5629a6b4b11d4a10d973aaf54d09c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261016T045710Z-9028-bece4d.log` | 4412 | 32 | `c27353123f4b0fa3a527646ed03bb21aaf2cb8d4b90e5d705567a0db4fab672d` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261016T105531Z-6792-e1a000.jsonl` | 113724 | 108 | `1700b4ea76d6f28e05eae534cee1c6b62bb67269cc0bf263e03cfe0352f36717` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261016T105531Z-6792-e1a000.log` | 16158 | 108 | `adc8432883413a3f1106ba84e81fdc7e0abf1013479146f0ab32d029f0120919` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261017T132214Z-9176-fe094e.jsonl` | 25147 | 28 | `60a47215330280e7e7d3e354775608cebe5c363f351a6e98bf7d01f1ca5e6128` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261017T132214Z-9176-fe094e.log` | 3769 | 28 | `4581aeffbf7be61999f5469b612ae0894f4b1cac55112a25f7768bb12a3e66ba` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261019T031447Z-8041-36ecd1.jsonl` | 33141 | 36 | `75ca312b7721b853fe652d7b62523f9ce6a9e85d801331a1c56be044754c788c` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261019T031447Z-8041-36ecd1.log` | 4854 | 36 | `34bfa77287df77e53b0bd697dbc85ba8bb7afc4503f748a756e38c7e46bd100b` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261019T092634Z-163180-c4dff3.jsonl` | 22723 | 20 | `459ed6af9c8a9e36fce3fcfc8a83d8c2eae23197de968c12335818fbe84f1894` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261019T092634Z-163180-c4dff3.log` | 2910 | 20 | `b02f28a3cb40b877f295bb1ad5f59bcbb9c00cd1cc685647b007266a024e14b5` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261020T051522Z-1675-9c8341.jsonl` | 28395 | 22 | `0a79f8ba7d8c7c9ae58f933a4b9c8e6ec50c5370d07a9bf87534c3b086c837ff` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261020T051522Z-1675-9c8341.log` | 3331 | 22 | `c8ef995b98112415f5df3ef7b7607ba66cb7801a78460fef45b47ee69c170731` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261021T033646Z-8491-328faa.jsonl` | 6103 | 11 | `7fce5d1e95a42b1cfffa8787a7601a1013119f472c540f7960ef4ce18741fc70` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261021T033646Z-8491-328faa.log` | 1119 | 11 | `dd821caa79a0c0730722a2a5db7892cea71c40404da3d2ac93fb6b2260050d78` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261021T050446Z-1713-afce2d.jsonl` | 24603 | 23 | `8244a7e893c4c7895b2b3e94071351efbfe68ab3611cd5440245885848776adc` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261021T050446Z-1713-afce2d.log` | 3308 | 23 | `36f01f3093d1117547aed96ab2c7a8c127cfc78fe057909b26edbc217224638d` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261026T033216Z-7618-d40d34.jsonl` | 8153 | 6 | `79823f41407e4e9f8d9089b2c66d0f82d0a8f0ee8d3bd6cc81a85d5015db74eb` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261026T033216Z-7618-d40d34.log` | 1249 | 6 | `b4ed3fc531fb2e2d4a8148ce00d97d9d9a4400856832e2db071be52a7e63f1fb` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261026T033246Z-7850-c03172.jsonl` | 26135 | 21 | `202c1c286891d12c707f21116d8e632aef77025f6899e7851905004b6795a592` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/20261026T033246Z-7850-c03172.log` | 3086 | 21 | `e8d80e764eabca4494269678c672c9871491577985083525f9b781a1166f7e88` | Training evidence; paired-run index and failure chronology |
| `output/logs/train/latest.json` | 2185 | 55 | `fa28a429e8f490beb2992c3f59559f3767ac030eef8aecc14426cff2a8d65dff` | Training evidence; paired-run index and failure chronology |
| `output/manifests/README.md` | 222 | 5 | `140828dd32a9b6c9ee8f8c255a6a5d560156e3d482cac33647b074b41680dbc6` | Pipeline metadata/log; parsed as supporting evidence |
| `output/manifests/cpu.jsonl` | 146107 | 282 | `1bc107470c6f5a7ff7c3503577497034a1b0e7a13b0690d2958fc47febb6d967` | Pipeline metadata/log; parsed as supporting evidence |
| `output/manifests/cpu.summary.json` | 1245 | 52 | `0ae5419a5f14b65b412c5fe09ab4bfb53ef14254e0ebeb1996d841d29984199a` | Pipeline metadata/log; parsed as supporting evidence |
| `output/manifests/cuda.summary.json` | 3611 | 128 | `49a5fedbee3d6f7899835973135463385bc75f6751b1cb039db0156d6370f73e` | Pipeline metadata/log; parsed as supporting evidence |
| `output/model/README.md` | 634 | 12 | `a842ac25402cddba388592285fcac08269c908d2d20df672821140176669a710` | Main metrics/config/status evidence; checkpoint tensors not supplied |
| `output/model/cuda-full/metrics.jsonl` | 276209 | 32 | `d24bf79fcbf0ac6ee9d58af87b9fb9a4f391299f104d11e9c2b8759efbb70f95` | Main metrics/config/status evidence; checkpoint tensors not supplied |
| `output/model/cuda-full/resolved_config.yaml` | 2161 | 116 | `636d3494c7406aa839c7ec2c30aacd433a8dcbb08d65bf9f1260b02b2ba0336c` | Main metrics/config/status evidence; checkpoint tensors not supplied |
| `output/model/cuda-full/status.json` | 2373 | 67 | `818034c88a5ef102ed457e3f7595fd976357ade5b4eba02d2bce1a437b03cd13` | Main metrics/config/status evidence; checkpoint tensors not supplied |
| `output/processed/README.md` | 246 | 5 | `b947a401031a85b31ef432a7b4e84bcd99e0a76aec8b3c368314a66c189a1ed5` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/README.md` | 432 | 9 | `cf5c2a1f48b60ba857de98b75a3033afde296f8245050fd631751763e82d57bf` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/doctor/cuda.json` | 5442 | 198 | `a1431c00d557eedc4bdd8a4140f92fb3bfe8d646bfedbe0315488dc84393a223` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-fcd.records.jsonl` | 1246000 | 2000 | `a9cece8fe440e9665c0ab6a767b0271581e481ed8222fcf294d69e66fde4fd57` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-fcd.state.json` | 418 | 11 | `94c4454a75e833bdf52de62626ab6fd3eca8f3f7a7c240b0e9c6b58fd373525c` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-scd.records.jsonl` | 11214000 | 18000 | `b045cab6bb6b9b1b7bcdde7eeb4281b89fd9e8301bebd4c9d761c93ab8bca162` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-scd.state.json` | 420 | 11 | `986a81f4a862281ee7f56b56c79b4eec0beb84a36360065b24261812dbe9a31f` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-testing.records.jsonl` | 19350000 | 30000 | `52083be50ab99359e9c1ed7013219bd76e6871a70803108dee24d78f1de13787` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-testing.state.json` | 432 | 11 | `ab32ea253b38a3205fed98dd9faeadb5ac1b3b4a7e6628bf6d34cf285df5cb2f` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-training.records.jsonl` | 77880000 | 120000 | `bbdf1d44503b3ec83f60c719df6e37d55ca400290e1b1998804429ca2d3e8889` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/doctamper-training.state.json` | 437 | 11 | `73b6f0c9c3af8c96382ffdd1f4a5ed6ea8831bce3908fd29949cb6c4c06d1687` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/midv-dm.records.jsonl` | 5154300 | 9000 | `6b1a780881db2b163be08f28beb1e2d9c5a0aaa722cd044c6a01c587d2ddb05c` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/midv-dm.state.json` | 400 | 11 | `1d5f4fe8d30b276d8e54d613541fb8a929d8be6daeef81a0e0466c5289ab3ac7` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/prepare/cuda/state.json` | 1341 | 31 | `32ed15e8e5998a954a1c90bb56750b8431b928259738933752845550f628201e` | Pipeline metadata/log; parsed as supporting evidence |
| `output/state/setup/cuda.json` | 410 | 11 | `1dec8fa64250e6c929c2c0c80693f6ef694ae0ecd31481425f4d5a665c7c106e` | Pipeline metadata/log; parsed as supporting evidence |
| `pyproject.toml` | 1351 | 50 | `9ce5c2d9f03d56b4b6ae8722667f76d841840621ca7affc7a5a102b1fb7b4794` | Documentation/configuration/metadata inventory; no repair mutation |
| `requirements-data.txt` | 30 | 2 | `fa1da675dda01f2400ddadc31d3c59f285b847133ed8ca0229859e24c87cbf83` | Documentation/configuration/metadata inventory; no repair mutation |
| `requirements-dev.txt` | 43 | 4 | `4697317b149a0cbb1b77177e6866d349074922899304199f43306f117184c0f8` | Documentation/configuration/metadata inventory; no repair mutation |
| `requirements.txt` | 49 | 5 | `9a3a5377805a2698e5e50121bf1b1bda9f0a1b6dc6916f6c55ceb03df575c1df` | Documentation/configuration/metadata inventory; no repair mutation |
| `setup/README.md` | 1606 | 30 | `16e1b326ef13066bb96f587a2f647209f3a506b251c5d7d2e99cc2fc7f32d2cf` | Environment/run instructions; checked against log runtime and resume command |
| `setup/REPRODUCIBILITY.md` | 1686 | 28 | `a59665b6d95410cdb9bbabfa4637c5881e6472616ba888653b6378d8f2a8e310` | Environment/run instructions; checked against log runtime and resume command |
| `setup/cpu.sh` | 5449 | 156 | `d3844397db0879cff9ceaf5dc475acd9dbc337aaeb8a928dd467a21e05fbe01b` | Environment/run instructions; checked against log runtime and resume command |
| `setup/cuda.sh` | 6706 | 186 | `76a95da68a373e1d5beca98bf59d96a86fab4dfabc0760dfcd5ee90a83456fa2` | Environment/run instructions; checked against log runtime and resume command |
| `setup/run.md` | 3356 | 75 | `8efbf10ce17a448e9877410e9bfabbea3a1a09647e66d17ce207ea4010708016` | Environment/run instructions; checked against log runtime and resume command |
| `setup/run.txt` | 3066 | 79 | `d21b24c7fcaae7b7b43cf5200925a4c7f0e44ba9ecf91076a5fd9dc73b178334` | Environment/run instructions; checked against log runtime and resume command |
| `setup/running` | 314 | — | `dddb45a14045c2f679a80742d0ce4dae4a13ca7d0d22f0ede8fa60843523ab1c` | Environment/run instructions; checked against log runtime and resume command |
| `temp.txt` | 823 | 54 | `7195f78c2943836ed88954d3db4858f30edc954669a6ff5036cea9c31ee29b1f` | Documentation/configuration/metadata inventory; no repair mutation |
| `tests/README.md` | 997 | 20 | `a62bca3fee51c9830907856659098c2aa31b60d167cd45e5e32d016b9f2f39c9` | Documentation/configuration/metadata inventory; no repair mutation |
| `tests/__pycache__/test_complete_model.cpython-312-pytest-9.1.1.pyc` | 27074 | — | `1e5d08133320dcafc923ac791e0a06de3eb042c28e4ea5ee3abc751f0a456f63` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_data_modelling.cpython-312-pytest-9.1.1.pyc` | 33782 | — | `1bbbf67241076276e1f6ef336a8733e0c0a362bc285ab54d2529947a78861968` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_degradation_estimator.cpython-312-pytest-9.1.1.pyc` | 25278 | — | `bf00c975bd5529199632a3ee3928799573660349d23b14e849604529e147db92` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_doctamper_converter.cpython-312-pytest-9.1.1.pyc` | 4843 | — | `77b71896a98303075343f1ad3d40d439b30b95b2ed16085b54bdf4e05192011a` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_freq_features.cpython-312-pytest-9.1.1.pyc` | 11401 | — | `2afef1e05c68ff55b06a7de8ab3db90cbb953c85f4f098bc8bd5728618a6cecb` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_fusion.cpython-312-pytest-9.1.1.pyc` | 21560 | — | `ab1de98cff947fe5eaeb3417d0efab6930d70776b1b4bbf396b9f760930724d0` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_manifest_dataset.cpython-312-pytest-9.1.1.pyc` | 20983 | — | `c1db71cca7f297fba88c4b6cc16deadf5abfdcc9cd22555ece80324812b0b825` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_postprocess_metrics.cpython-312-pytest-9.1.1.pyc` | 9484 | — | `9542fdf8cabf792fac94e407237930460f7dea2f39d87ce703ca30a1b16ffd2d` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_prepare_lmdb_v04.cpython-312-pytest-9.1.1.pyc` | 5837 | — | `58c4af3d073453d4cb1022a131ead69c5482a2f2430b2de3db4bc512e79bd8f1` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_protocol_v04.cpython-312-pytest-9.1.1.pyc` | 25919 | — | `71df629ee181eb281e26e20886a8544026884581be4014320953dd35135602bd` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_regressions_v04.cpython-312-pytest-9.1.1.pyc` | 22592 | — | `9aef18aa3c32639471386078ba1971f481ff213eb4b296fb7222725e4ed0a3b8` | Compiled cache; inventoried only; excluded from repair |
| `tests/__pycache__/test_spatial_features.cpython-312-pytest-9.1.1.pyc` | 15097 | — | `b2dac0c20adfd715e8b828c52e217b5601355f6acdd05f35926823e198be1799` | Compiled cache; inventoried only; excluded from repair |
| `tests/test_complete_model.py` | 5814 | 135 | `686d07061c28a462ce50d3967f2d4116e557d6fbab1b5caea02a56969ad6c560` | Regression source; included in executed suite |
| `tests/test_data_modelling.py` | 4745 | 122 | `a2e98eff1584f9c6f65a69c4bf548f7b553e4844dbdb27a55c2d00fea6b82254` | Regression source; included in executed suite |
| `tests/test_degradation_estimator.py` | 4322 | 117 | `235cea738620e976c870a34b4fc9fecf637a335a52a0d0c559df38c16f06176c` | Regression source; included in executed suite |
| `tests/test_doctamper_converter.py` | 954 | 31 | `2a72d54c9877b02063e11f95e6a89084d03151c4d074be35b7f80d3c4bc59a14` | Regression source; included in executed suite |
| `tests/test_freq_features.py` | 3426 | 102 | `2369171c417f5acdc77d45220de8a63b13aabf9be636e5543bb3783052412fc7` | Regression source; included in executed suite |
| `tests/test_fusion.py` | 5863 | 159 | `bc8855d85e6abd80f8144a7d9e8e596bfe5edd050a22cc532bda3bf10466d6fb` | Regression source; included in executed suite |
| `tests/test_manifest_dataset.py` | 6504 | 190 | `825814f052f98cfe2b0d8f151c288dd1021ba42e69c91c1aaef627b3fdda0745` | Regression source; included in executed suite |
| `tests/test_postprocess_metrics.py` | 2067 | 58 | `f16eab9ec6467528ea20d0f232b5fd6a36a8073215fe81d895f29725870f2456` | Regression source; included in executed suite |
| `tests/test_prepare_lmdb_v04.py` | 1596 | 53 | `62d821f6dfec6c02339a36c480f0ca07eddb5bf1d458ffa851cfe13cf546fee6` | Regression source; included in executed suite |
| `tests/test_protocol_v04.py` | 13708 | 387 | `0c04f0e5d09f2dc3023b2c7ab4e3bb3e45bd46c85d357667ba707c08832a351f` | Regression source; included in executed suite |
| `tests/test_regressions_v04.py` | 7258 | 169 | `ce1a02bc5dfff9b488f84faa34d467f6a6016fd872fa082a5875169d34076e10` | Regression source; included in executed suite |
| `tests/test_regressions_v053.py` | 3439 | 91 | `61aadc9e77ff4d967ae3390268e4ea7ee2d8e9e6d4360c350e7f6dc45b204304` | Regression source; included in executed suite |
| `tests/test_resume_v051.py` | 4832 | 143 | `0ce50d6e2846c7f3bccc083d9da0282174b8d4897c48543079db31b1ab172045` | Regression source; included in executed suite |
| `tests/test_spatial_features.py` | 3399 | 101 | `27523e6aedb8dce75ee19428743931bcb3b55be6a64666deeccf591efee1bcc1` | Regression source; included in executed suite |
| `tests/test_telemetry.py` | 4876 | 136 | `cf51ec5730bdf5e1aec1220b6dee9dea60b4bf1f2165b7915643eea397638a84` | Regression source; included in executed suite |
| `workflow.jpeg` | 97614 | — | `928dd6b59f4310a1b4f5f60bb4c274b646be60708d3a643920d837482e6f4254` | Binary illustration/asset; inventoried; not used to infer runtime behaviour |
