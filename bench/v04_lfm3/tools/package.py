#!/usr/bin/env python3

import zipfile
from common import ROOT, BUILD, MODEL


def main():
    BUILD.mkdir(exist_ok=True)
    dest = BUILD / "lfm3.pk3"
    temp = BUILD / "lfm3.tmp.pk3"
    with zipfile.ZipFile(temp, "w", allowZip64=True) as archive:
        for path in sorted((ROOT / "src").rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(ROOT / "src").as_posix(),
                    compress_type=zipfile.ZIP_STORED
                    if path.suffix in (".bin", ".scale")
                    else zipfile.ZIP_DEFLATED,
                )
        archive.write(MODEL / "LICENSE", "LFM-LICENSE.txt")
        archive.write(ROOT / "assets.lock.json", "assets.lock.json")
    temp.replace(dest)
    print(dest, dest.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
