"""Prepare the public MiniT2I datasets; network work requires --download.

The small REMOTE_LOCK below is a 2026-10-16 metadata snapshot, not data.
Dataset transforms follow the official MiniT2I preparation scripts at
d92730b2a01667a07023ff9104b9baab29b4030f (MIT). See t2i/README.md for the
DALL-E source discrepancy and differences needed for fail-closed preparation.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from .data import expand_shards, shard_identity
from .assets import T5_MANIFEST_FORMAT, T5_SOURCE_LOCK, validate_t5_assets


REMOTE_LOCK = {'cc12m': {'repo_id': 'CaptionEmporium/conceptual-captions-cc12m-llavanext',
           'repo_type': 'dataset',
           'revision': '1c24bdba3f81257082612654ba72fe18a83c7a37',
           'files': [{'name': 'train.jsonl.gz',
                      'size': 3577232003,
                      'sha256': 'da25ecab422be301215c8d7e3bd55a795375f036c1d0b7d250e0f59c9a8b284a'}]},
 'blip3o': {'repo_id': 'BLIP3o/BLIP3o-60k',
            'repo_type': 'dataset',
            'revision': 'f7316b0aa446338ee1707484924aa59457b4bbf3',
            'files': [{'name': 'dalle3.tar',
                       'size': 10566379520,
                       'sha256': 'cf05ef7eb54fccb2da0042f80d0c93d31a3aaff82d034b6ff063ea3dc6f76afa'},
                      {'name': 'geneval_train.tar',
                       'size': 12248104960,
                       'sha256': '2ac91805708111132c7bfa06ca51eef1ba9a4704b364e8996ae10fde13898a54'},
                      {'name': 'human_gestures.tar',
                       'size': 17537228800,
                       'sha256': 'f951b8d759396ba6a3ba27e90aa3c4fee0a260da2e670e559d8cb3ac064fa1f3'},
                      {'name': 'journeyDB.tar',
                       'size': 18301818880,
                       'sha256': '6c8f2019796b33b0c6944831fb470398104188444cccf201cf5684d39c253358'},
                      {'name': 'mscoco_human.tar',
                       'size': 15540961280,
                       'sha256': '69b942314ca631b702872bd8e942a86667e006519d310f39c442f0da2b5430ee'},
                      {'name': 'object_1.tar',
                       'size': 4509962240,
                       'sha256': '2a61ccc55af55f5c2ad4afb5d5f8e52430900b50607c52a5b7672bac305cf1ab'},
                      {'name': 'object_2.tar',
                       'size': 1347297280,
                       'sha256': 'd32f59d17e282dfebaef464844d39df8d5c31bf9ee73e51809fe25594f7913ba'},
                      {'name': 'occupation_1.tar',
                       'size': 4462090240,
                       'sha256': '29ad915f8a965966471214a9a91ca72ba987f55c55d445b57dbaeaf2cdbe45c6'},
                      {'name': 'occupation_2.tar',
                       'size': 4457164800,
                       'sha256': 'aa3cb471a653ababdd897a291f0fdae9a5a6eec46b546a60f3de70fbfc10c8de'},
                      {'name': 'text_1.tar',
                       'size': 9582243840,
                       'sha256': '9c655accc7cad2a6af41ff56e7d0668560e3965766eb743e07c5973bf9a84fc0'},
                      {'name': 'text_2.tar',
                       'size': 9607751680,
                       'sha256': '122ab505623f912bf642ddc5dad989e5cbee969d7de5312ef00ae5685b4cb430'}]},
 'dalle3': {'repo_id': 'OpenDatasets/dalle-3-dataset',
            'repo_type': 'dataset',
            'revision': '96822d5c0b1338ec4bfd04b0d5909fa070e301b6',
            'files': [{'name': 'data/train-00000-of-00067.parquet',
                       'size': 513914389,
                       'sha256': '42f814346a3576292ca388e127202975ef85dc6e8dec47bed612cbe7287d93a0'},
                      {'name': 'data/train-00001-of-00067.parquet',
                       'size': 445430025,
                       'sha256': 'a3d5116b15af94b0b57726f47b4d05204be138fb4ba926f50d9f2263f3a9d77d'},
                      {'name': 'data/train-00002-of-00067.parquet',
                       'size': 447936336,
                       'sha256': '8926a5a0cdb097401b1e19d89839b42f50bb250196ebe2e485dfa3955dd7d87b'},
                      {'name': 'data/train-00003-of-00067.parquet',
                       'size': 406427498,
                       'sha256': '3e6e9991b30f094575a7000fffda67c40a2fcd451cc88c7128201d6ff65dd8b2'},
                      {'name': 'data/train-00004-of-00067.parquet',
                       'size': 473643230,
                       'sha256': 'b49fb524d52bf9857d15fb88b93e2c41795655ae0c55d222854d6989b3bbbb3d'},
                      {'name': 'data/train-00005-of-00067.parquet',
                       'size': 658988916,
                       'sha256': 'b9f093d4a33df91f993cf8a14e7ced8ac700d5ab91272acbcc3642b5d36394a2'},
                      {'name': 'data/train-00006-of-00067.parquet',
                       'size': 401865272,
                       'sha256': 'f7e35d6442c0eb932ce9b4ef257d8af7f9cf5763b710ea5f99f3db1c67651b86'},
                      {'name': 'data/train-00007-of-00067.parquet',
                       'size': 508439321,
                       'sha256': 'f893da83fa840471a4bc442ed0dc6e98c541ddb4d5eeaafa27d955c1af2924a7'},
                      {'name': 'data/train-00008-of-00067.parquet',
                       'size': 470608634,
                       'sha256': '1cde91b8cc82aed5f6f461aa6d740dfbd8276160d923383e48d9312e271f9eae'},
                      {'name': 'data/train-00009-of-00067.parquet',
                       'size': 416996902,
                       'sha256': 'cdd4fe06576090542ce0219ec7139d57dff7882a0ae005451b35e52f68a02b17'},
                      {'name': 'data/train-00010-of-00067.parquet',
                       'size': 398940343,
                       'sha256': 'a3c4d4befd6e2f1a4ae24db714f826de9d7769b41369308f96d905601e81b5a0'},
                      {'name': 'data/train-00011-of-00067.parquet',
                       'size': 466673434,
                       'sha256': 'bee0e35b26e8b443ee3cb577664ba3982e4e39e78e9928dfbeadac388d4d90df'},
                      {'name': 'data/train-00012-of-00067.parquet',
                       'size': 383881618,
                       'sha256': '473c7c140eeff5739928fb7807c1ea90e63ac846e276c9bb387418177b639c8b'},
                      {'name': 'data/train-00013-of-00067.parquet',
                       'size': 461414060,
                       'sha256': 'a884634e3c5e0ee6e6da9e808cff2a97787bb1183c6f75139e6c40b7ff3a7314'},
                      {'name': 'data/train-00014-of-00067.parquet',
                       'size': 557451878,
                       'sha256': '7520fa5d9e0482c6c462bdacadf1bbf9c7372fa7f6110692a07f7092cd12c9ac'},
                      {'name': 'data/train-00015-of-00067.parquet',
                       'size': 446241182,
                       'sha256': '1498e8c51a2158c826f339e18b58b154173aeb77d7de59c9a8687b78f73e4baf'},
                      {'name': 'data/train-00016-of-00067.parquet',
                       'size': 563065114,
                       'sha256': 'c745c0f11d1f1e236c284e3dfaff105121cf2fd97b65ca2b7daf7d21be3d3089'},
                      {'name': 'data/train-00017-of-00067.parquet',
                       'size': 542866314,
                       'sha256': '38bba172ef4357f375b297861cb8ab8135fd0893d9a899897dac9717280ada90'},
                      {'name': 'data/train-00018-of-00067.parquet',
                       'size': 490748672,
                       'sha256': '99518f4416a56fb63d123516f4ebd72cb2d8b7414ad0419caf253ec1f4cda11a'},
                      {'name': 'data/train-00019-of-00067.parquet',
                       'size': 539392411,
                       'sha256': '3b815101afad9124a719c822f23f547b5f9c2de35460ec9c11e2a5b7eede1437'},
                      {'name': 'data/train-00020-of-00067.parquet',
                       'size': 454923221,
                       'sha256': 'a131d1e1d77da47904c355356b4aa574ac4e88d8965864064bd45194ec6a4792'},
                      {'name': 'data/train-00021-of-00067.parquet',
                       'size': 534068297,
                       'sha256': 'ff47d4b75e8f9f550d07a16a413f255cbc2da5188da77482c55f6a3246deb485'},
                      {'name': 'data/train-00022-of-00067.parquet',
                       'size': 486880449,
                       'sha256': '923c43b0e3c036c4b90e50606cb911fd073c8e0dee50fdd38707047e13b8e94e'},
                      {'name': 'data/train-00023-of-00067.parquet',
                       'size': 526548269,
                       'sha256': '6de34de2161730eaf6bf1fe6c5a3c499b804d747bacfbdff4972fd88bba4d115'},
                      {'name': 'data/train-00024-of-00067.parquet',
                       'size': 444935347,
                       'sha256': 'f83c26329603171a423357d1320265c521bce7f7edbfffaf2f41fee2cdbb470f'},
                      {'name': 'data/train-00025-of-00067.parquet',
                       'size': 449943178,
                       'sha256': '7ff141ae034e8b28e567511ad146cf8b37d13eba210698d5c76c0068b137d711'},
                      {'name': 'data/train-00026-of-00067.parquet',
                       'size': 480889212,
                       'sha256': '397b4f4e33cec5b835d83023b054956ca90dabde99c3513419499bc28e9b9f58'},
                      {'name': 'data/train-00027-of-00067.parquet',
                       'size': 518480577,
                       'sha256': '539a7e438d91051ecb0246c607b355692f9fef7b05f68b20ff0b3a504bbf4947'},
                      {'name': 'data/train-00028-of-00067.parquet',
                       'size': 560188844,
                       'sha256': '9027ee17a56f492449792b1a64f728902b9665de0a270a1efb2727cad245bd8b'},
                      {'name': 'data/train-00029-of-00067.parquet',
                       'size': 488351530,
                       'sha256': 'e012535b1245063c541998916a799eac13c1cdc5aaceb7f2c1a02ad1d7a5b6b6'},
                      {'name': 'data/train-00030-of-00067.parquet',
                       'size': 470684944,
                       'sha256': '762aab3fb255aacb825a43c7cb2992994fbf8b1c7b9ce49db8621c2460d1f9d9'},
                      {'name': 'data/train-00031-of-00067.parquet',
                       'size': 539744503,
                       'sha256': 'faca4cd698d90157ce88f7732415299a5058829858abf22c61d14b1a6307dc9b'},
                      {'name': 'data/train-00032-of-00067.parquet',
                       'size': 472069971,
                       'sha256': '21670dc3c86efc050b9c2a19b1e60791a73bd962987e7b28f0f5e0d186fe8b9a'},
                      {'name': 'data/train-00033-of-00067.parquet',
                       'size': 645404086,
                       'sha256': '62075a7908332724481a399e84ec91a4fe6585c1d4724ccef4d66149b9584fb5'},
                      {'name': 'data/train-00034-of-00067.parquet',
                       'size': 587433061,
                       'sha256': '8f27be18e864c6e65b1359e474a74189e64e44d6ca61a1dc68e11f45cd6f8ee9'},
                      {'name': 'data/train-00035-of-00067.parquet',
                       'size': 564783659,
                       'sha256': '57086fbc4e51013b25aba09f046bdee682621a094d16eccf83fee7ae39e4033e'},
                      {'name': 'data/train-00036-of-00067.parquet',
                       'size': 623343516,
                       'sha256': '210590c31a2dde0e460c9479e33c31b9fa914b645cb8939daae7d4422e6bab90'},
                      {'name': 'data/train-00037-of-00067.parquet',
                       'size': 564972720,
                       'sha256': '2f8b51a18ee591b5101961dbfea3eee4a5c2369d985050a5a819913001932971'},
                      {'name': 'data/train-00038-of-00067.parquet',
                       'size': 450807193,
                       'sha256': 'f6340cdecb87b4f272962afc7b6bcaac23014785ed2120b324787a9b48e9df84'},
                      {'name': 'data/train-00039-of-00067.parquet',
                       'size': 423138479,
                       'sha256': '6dde2482c07db1295348000cf02acde8d4be503534440db8e1ab10ec543eeebc'},
                      {'name': 'data/train-00040-of-00067.parquet',
                       'size': 465841798,
                       'sha256': 'd4d48a90b749a4cd606dcf9f5fc08f66c45b2f43b42feeb100ceb391a4104062'},
                      {'name': 'data/train-00041-of-00067.parquet',
                       'size': 423055488,
                       'sha256': '95d803ecec063d375de385349c6c6408904c19904c77209fcbc606598b8787b0'},
                      {'name': 'data/train-00042-of-00067.parquet',
                       'size': 410571171,
                       'sha256': '2606908058773a883ee7ee0f9844a05813959b7c4d005fb5c1cc96997966b29a'},
                      {'name': 'data/train-00043-of-00067.parquet',
                       'size': 380338564,
                       'sha256': '9162a08b8d51aae84fd53eba4ed6e2732a435c19358a8999ff9663199278fa0a'},
                      {'name': 'data/train-00044-of-00067.parquet',
                       'size': 496308683,
                       'sha256': '55c2a47d79cd39acb2c836ab61f66b5ae77725b6e177f009ad1fa51e9b4eba94'},
                      {'name': 'data/train-00045-of-00067.parquet',
                       'size': 482136753,
                       'sha256': 'a3e8f042d8951df4d9ce528c95aa76b3fb935d8b51f5f64cc885b82bbfb17fdd'},
                      {'name': 'data/train-00046-of-00067.parquet',
                       'size': 539717539,
                       'sha256': '9acccacca842f1a460415aa9c359188c23c007e485f4d6a3c44827df0181b0ee'},
                      {'name': 'data/train-00047-of-00067.parquet',
                       'size': 529928518,
                       'sha256': '260f52700be31923935d3bf98b2a07d1e87db168947c4fc9a733c12c64598ce1'},
                      {'name': 'data/train-00048-of-00067.parquet',
                       'size': 518359675,
                       'sha256': '206c34c280e4ceb5765d08cd51f88ec7b3c010d3efb5ba7e7c966475b9d53b2f'},
                      {'name': 'data/train-00049-of-00067.parquet',
                       'size': 478702434,
                       'sha256': 'fa9f96e0fe5006ac83b35441f4ae93527e66b766d084b0025f9aa5949bc692a7'},
                      {'name': 'data/train-00050-of-00067.parquet',
                       'size': 471638975,
                       'sha256': 'f7cdf5a5ff74a4c456a1c778c6ed5fb3e36769fde0fbd18351d9b1868fb66566'},
                      {'name': 'data/train-00051-of-00067.parquet',
                       'size': 447018831,
                       'sha256': '3d6f28637bed446d617aeca16ad9c2e8d9c84e5d5e9a385ac1857176746a0270'},
                      {'name': 'data/train-00052-of-00067.parquet',
                       'size': 575143957,
                       'sha256': 'e4a63cdfb6c59f46f34ee66f5b3bdad018da5de42f5fbac99f8036122dda98ba'},
                      {'name': 'data/train-00053-of-00067.parquet',
                       'size': 593602037,
                       'sha256': 'aeba3aec83a079e94293a06b8b831acf4cb494d30a43122ead978d73afdaaec7'},
                      {'name': 'data/train-00054-of-00067.parquet',
                       'size': 454347582,
                       'sha256': '0b411e149da5a89d1e9ddc84554c73e89be71986f28b7a24e6216790a48a31fc'},
                      {'name': 'data/train-00055-of-00067.parquet',
                       'size': 499648943,
                       'sha256': '8a5f6f3e060d297cf758b7e63eb457e5d87b59f7b0064ec588e6f6cc46cff4b4'},
                      {'name': 'data/train-00056-of-00067.parquet',
                       'size': 403124685,
                       'sha256': '0ea3c92b45011a93dfe5930ebf6b1195161a35ae93f2dcde24b9f1f24d82fcb1'},
                      {'name': 'data/train-00057-of-00067.parquet',
                       'size': 475768061,
                       'sha256': 'c6687ac064f1c82933a88410c2dc4f089772e40a78f82bf3ec13623c2be0dca2'},
                      {'name': 'data/train-00058-of-00067.parquet',
                       'size': 627163163,
                       'sha256': '0017e8c0c0872dee55dc338578c8fcb0a2737d573fd97f139543282e5e86703e'},
                      {'name': 'data/train-00059-of-00067.parquet',
                       'size': 519753892,
                       'sha256': '50574a0fb413cbab029562822ca8612e00f5d8c6da6028fc54da8b6243095227'},
                      {'name': 'data/train-00060-of-00067.parquet',
                       'size': 439615370,
                       'sha256': '41d7d898f41ce578bb5c3b1abc0f0e7bbd0964dd2e64145ad5c9b05eb0c7186d'},
                      {'name': 'data/train-00061-of-00067.parquet',
                       'size': 476701839,
                       'sha256': 'aad10af402b2101e1c9c5be3d3c704ceb3083e27e6fd8cf05be70310ca6e6796'},
                      {'name': 'data/train-00062-of-00067-49cc56f79f975fae.parquet',
                       'size': 444570886,
                       'sha256': '3435c52a882e2223591ab59ec2e167780b85e68eac9778f6cc0f39c27488b0f6'},
                      {'name': 'data/train-00063-of-00067-c51100b2304548b8.parquet',
                       'size': 453890844,
                       'sha256': '6b302e2a618ba532a469441efcaeb382bd7b26eefe454ba8956f6a294cd237e2'},
                      {'name': 'data/train-00064-of-00067-31b9925885a214be.parquet',
                       'size': 498429665,
                       'sha256': '9086c5844ada70a06fda2879334fe37ac60f23017f5de092de4d98d71f02ca9f'},
                      {'name': 'data/train-00065-of-00067-0aab86a82bda475a.parquet',
                       'size': 606949829,
                       'sha256': '4418bed0f3d7540ce11e5a1293f40861df7d444cdffe805f47448fefebe7c001'},
                      {'name': 'data/train-00066-of-00067-31a40dd1119835ff.parquet',
                       'size': 591065501,
                       'sha256': '704159f23372c11bee700864fb9b825319a35b954d189ea67f6601b324933112'}]},
 'sharegpt4o': {'repo_id': 'FreedomIntelligence/ShareGPT-4o-Image',
                'repo_type': 'dataset',
                'revision': 'f9bcb9494e562f3b6d46585bda0e7f0f0d6ec43f',
                'files': [{'name': 'text_to_image.json',
                           'size': 28083319,
                           'sha256': 'daeea6b4fb22eab3164b96687acb85ead590eafb3cbbee1b3163f393396910d2'},
                          {'name': 'text_to_image_part_0.tar',
                           'size': 9271982080,
                           'sha256': '478bd64349c8ba353ef96542999363054e79599b17ec84df7c545eeb680011f8'},
                          {'name': 'text_to_image_part_1.tar',
                           'size': 9207316480,
                           'sha256': '7721137cbadabf497c6cc775665ed0c3b38592e8c0fd679b02d431a10a975c24'},
                          {'name': 'text_to_image_part_2.tar',
                           'size': 9289912320,
                           'sha256': '42e30858915b0f4370fbbd1e7fc481df41d5e9c9489d2d049e23cf75bc6e0955'},
                          {'name': 'text_to_image_part_3.tar',
                           'size': 9295093760,
                           'sha256': 'cac143c0a2452e0bced59b5fa19e4fcbef8ef9426367c3114256b44a66a01f7c'},
                          {'name': 'text_to_image_part_4.tar',
                           'size': 8873216000,
                           'sha256': '0a9454162516d2dc1adcb54977818c14f6f87c68b7f33e30f6cc976f17867b42'},
                          {'name': 'text_to_image_part_5.tar',
                           'size': 8237957120,
                           'sha256': '2ab345ed3b7f0848debd29f60caf71e3973af2766e52119e6d83455b629f4772'},
                          {'name': 'text_to_image_part_6.tar',
                           'size': 8428677120,
                           'sha256': 'b5ac290f4a59fc84ae1fbbe1452230909d4683cda62a293a9dff6f0af7cd86f3'},
                          {'name': 'text_to_image_part_7.tar',
                           'size': 8996177920,
                           'sha256': '7eef2cea109700c8bb1e5f12b5b769a4edc171c91686c479d992594ee7d98149'},
                          {'name': 'text_to_image_part_8.tar',
                           'size': 9190768640,
                           'sha256': '5bc22d550a6f5ead80edf4c336f1cf3e12ed73a3200acf6042ca19839265275f'},
                          {'name': 'text_to_image_part_9.tar',
                           'size': 9283911680,
                           'sha256': 'e081a6aa8f0601c9ee149b9be6eaa324371077f9b298760ded712cca4197768c'}]},
 't5': T5_SOURCE_LOCK}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value, *, create_once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_once and path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"Refusing to replace an existing preparation receipt: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if create_once:
        os.link(name, path)
        # Only this call's temporary file is removed.
        os.unlink(name)
    else:
        os.replace(name, path)


def download_group(key, root):
    from huggingface_hub import hf_hub_download
    lock, root = REMOTE_LOCK[key], Path(root)
    actual = {}
    for entry in lock["files"]:
        target = root / entry["name"]
        if not target.exists():
            target = Path(hf_hub_download(lock["repo_id"], entry["name"],
                                         repo_type=lock["repo_type"], revision=lock["revision"],
                                         local_dir=str(root)))
        digest = sha256(target)
        verdict = {"repo_id": lock["repo_id"], "revision": lock["revision"],
                   "file": entry["name"], "expected_size": entry["size"],
                   "actual_size": target.stat().st_size, "expected_sha256": entry["sha256"],
                   "actual_sha256": digest}
        verdict["pass"] = verdict["actual_size"] == entry["size"] and digest == entry["sha256"]
        # The exact values used by this gate survive failures.
        atomic_json(root / "download.progress.json", verdict)
        if not verdict["pass"]:
            raise ValueError(f"Pinned download failed checksum/size validation: {target}")
        actual[entry["name"]] = digest
    binding = {"files": actual, "source": lock}
    if key == "t5":
        binding.update(format=T5_MANIFEST_FORMAT, checkpoint_format="flax", from_pt=False)
    atomic_json(root / "assets.sha256.json", binding, create_once=True)
    return root


class PairWriter:
    """Create new flat tars only; never truncate an existing output shard."""
    def __init__(self, root, shard_size=3000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size, self.total, self.part = int(shard_size), 0, -1
        self.tar, self.files = None, {}

    def add(self, key, image, caption):
        if self.total % self.shard_size == 0:
            self.close()
            self.part += 1
            path = self.root / f"shard-{self.part:05d}.tar"
            self.tar = tarfile.open(fileobj=path.open("xb"), mode="w")
            self._path = path
        for ext, data in (("jpg", image), ("txt", (caption.strip() + "\n").encode())):
            member = tarfile.TarInfo(f"{key}.{ext}")
            member.size = len(data)
            self.tar.addfile(member, io.BytesIO(data))
        self.total += 1

    def close(self):
        if self.tar is not None:
            # TarFile does not close caller-owned file objects.
            fileobj = self.tar.fileobj
            self.tar.close()
            fileobj.close()
            self.files[self._path.name] = sha256(self._path)
            self.tar = None

    def finish(self, source):
        self.close()
        if not self.total:
            raise ValueError("Dataset conversion produced no image/caption pairs")
        atomic_json(self.root / "preparation.json", {"samples": self.total, "files": self.files,
                                                     "source": source}, create_once=True)


def _completed(root):
    receipt = Path(root) / "preparation.json"
    if not receipt.exists():
        if Path(root).exists() and any(Path(root).glob("*.tar")):
            raise FileExistsError(f"Incomplete pre-existing output requires inspection: {root}")
        return False
    value = json.loads(receipt.read_text())
    for name, expected in value["files"].items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"Prepared dataset checksum changed: {name}")
    return True


def prepare_cc12m(data_dir, processes, threads):
    raw = download_group("cc12m", data_dir / "raw" / "cc12m")
    output = data_dir / "cc12m"
    if _completed(output):
        return
    parquet_root = raw / "parquet"
    parquet_root.mkdir(exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Chunked conversion keeps metadata RAM bounded; the long caption is the
    # actual upstream `caption_llava`, not original alt-text or short caption.
    with gzip.open(raw / "train.jsonl.gz", "rt") as reader:
        rows, part = [], 0
        for line in reader:
            row = json.loads(line)
            if row.get("status", "success") != "success":
                continue
            rows.append({"url": row["url"], "caption": row["caption_llava"], "key": row["key"]})
            if len(rows) == 10000:
                target = parquet_root / f"part-{part:05d}.parquet"
                if target.exists():
                    raise FileExistsError(f"Incomplete metadata conversion: {target}")
                pq.write_table(pa.Table.from_pylist(rows), target)
                rows, part = [], part + 1
        if rows:
            target = parquet_root / f"part-{part:05d}.parquet"
            if target.exists():
                raise FileExistsError(target)
            pq.write_table(pa.Table.from_pylist(rows), target)
    # Official JAX script uses resize_mode=no, not pre-cropped latent images.
    # setup.sh uses its venv Python without activating PATH. Keep this lexical
    # parent: resolving a venv's Python symlink would escape the environment.
    img2dataset = Path(sys.executable).parent / "img2dataset"
    subprocess.run([str(img2dataset), "--url_list", str(parquet_root), "--input_format", "parquet",
                    "--url_col", "url", "--caption_col", "caption", "--output_format", "webdataset",
                    "--output_folder", str(output), "--image_size", "512", "--resize_mode", "no",
                    "--processes_count", str(processes), "--thread_count", str(threads),
                    "--number_sample_per_shard", "10000", "--enable_wandb", "False"], check=True)
    files = {p.name: sha256(p) for p in expand_shards(output)}
    atomic_json(output / "preparation.json", {"files": files, "source": REMOTE_LOCK["cc12m"],
                                            "note": "URL availability changes realized sample count; inspect img2dataset stats."},
                create_once=True)


def prepare_dalle3(data_dir):
    output = data_dir / "curated" / "dalle3"
    if _completed(output):
        return
    raw = download_group("dalle3", data_dir / "raw" / "dalle3")
    import pyarrow.parquet as pq
    writer = PairWriter(output)
    try:
        index = 0
        for path in sorted((raw / "data").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    image = row["image"]
                    if isinstance(image, dict):
                        image = image.get("bytes")
                    if isinstance(image, str):
                        image = base64.b64decode(image.split(",", 1)[-1])
                    if not isinstance(image, (bytes, bytearray)):
                        raise ValueError(f"Unsupported DALL-E image encoding at row {index}")
                    writer.add(f"{index:012d}", bytes(image), row["caption"])
                    index += 1
        writer.finish(REMOTE_LOCK["dalle3"])
    finally:
        writer.close()


def prepare_sharegpt4o(data_dir, t5_model):
    output = data_dir / "curated" / "sharegpt4o"
    if _completed(output):
        return
    raw = download_group("sharegpt4o", data_dir / "raw" / "sharegpt4o")
    validate_t5_assets(t5_model, verify_payloads=False)
    from transformers import AutoTokenizer
    from PIL import Image
    tokenizer = AutoTokenizer.from_pretrained(str(t5_model), local_files_only=True)
    captions = {}
    for row in json.loads((raw / "text_to_image.json").read_text()):
        image = row.get("output_image", "")
        stem = image.removeprefix("image/").removesuffix(".png")
        caption = row.get("input_prompt", "")
        if image.startswith("image/") and image.endswith(".png") and stem.isdigit():
            if len(tokenizer(caption, truncation=False)["input_ids"]) <= 256:
                captions[int(stem)] = caption
    writer, seen = PairWriter(output), set()
    try:
        for path in sorted(raw.glob("text_to_image_part_*.tar")):
            with tarfile.open(path, "r|*") as source:
                for member in source:
                    source.members.clear()
                    stem = member.name.removeprefix("image/").removesuffix(".png")
                    if not member.isfile() or not member.name.endswith(".png") or not stem.isdigit():
                        continue
                    index = int(stem)
                    if index not in captions:
                        continue
                    if index in seen:
                        raise ValueError(f"Duplicate ShareGPT image id: {index}")
                    with Image.open(io.BytesIO(source.extractfile(member).read())) as image:
                        target = io.BytesIO()
                        image.convert("RGB").save(target, format="JPEG", quality=95)
                    writer.add(str(index), target.getvalue(), captions[index])
                    seen.add(index)
        verdict = {"expected_filtered_captions": len(captions), "written_images": len(seen),
                   "missing_images": sorted(set(captions) - seen)}
        atomic_json(output / "conversion.progress.json", verdict)
        if verdict["missing_images"]:
            raise ValueError("ShareGPT images referenced by filtered metadata are missing")
        writer.finish({"dataset": REMOTE_LOCK["sharegpt4o"], "max_t5_tokens": 256,
                       "jpeg_quality": 95, "tokenizer": REMOTE_LOCK["t5"]["revision"]})
    finally:
        writer.close()


def prepare_curated(data_dir, t5_model):
    download_group("blip3o", data_dir / "curated" / "blip3o")
    prepare_dalle3(data_dir)
    prepare_sharegpt4o(data_dir, t5_model)
    atomic_json(data_dir / "curated" / "sources.json", {
        "description": "MiniT2I public '120K' mixture; approximate name, not a cardinality assertion",
        "upstream_revision": "d92730b2a01667a07023ff9104b9baab29b4030f",
        "sources": [{"name": "blip3o", "root": "blip3o", "weight": 0.06},
                    {"name": "dalle3", "root": "dalle3", "weight": 0.016},
                    {"name": "sharegpt4o", "root": "sharegpt4o", "weight": 0.04}]}, create_once=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "t5", "datasets", "check"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--t5-model", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cc12m-root", default="")
    parser.add_argument("--curated-root", default="")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    t5_model = (args.t5_model or data_dir / "assets" / "t5").resolve()
    if args.command == "plan":
        print(json.dumps({key: {"repository": lock["repo_id"], "revision": lock["revision"],
                                "download_bytes": sum(f["size"] for f in lock["files"])}
                          for key, lock in REMOTE_LOCK.items()}, indent=2))
    elif args.command == "check":
        for root in (args.cc12m_root, args.curated_root):
            if not root:
                raise ValueError("Both CC12M_ROOT and CURATED_ROOT must be set")
            path = Path(root)
            spec_path = path / "sources.json" if path.is_dir() and (path / "sources.json").is_file() else path
            if spec_path.suffix == ".json":
                spec = json.loads(spec_path.read_text())
                identities = [shard_identity(p) for source in spec["sources"]
                              for p in expand_shards(str(spec_path.parent / source["root"]))]
            else:
                identities = [shard_identity(p) for p in expand_shards(root)]
            print(json.dumps({"root": root, "shards": len(identities),
                              "bytes": sum(p["size"] for p in identities),
                              "identity_sha256": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()}))
    else:
        if not args.download:
            parser.error("Network preparation requires explicit --download; use plan to inspect sizes")
        if args.command == "t5":
            download_group("t5", t5_model)
        else:
            if args.processes < 1 or args.threads < 1:
                parser.error("Download process/thread counts must be positive")
            if not args.cc12m_root:
                prepare_cc12m(data_dir, args.processes, args.threads)
            if not args.curated_root:
                prepare_curated(data_dir, t5_model)


if __name__ == "__main__":
    main()
