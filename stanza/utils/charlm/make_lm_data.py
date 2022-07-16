"""
Create Stanza character LM train/dev/test data, by reading from txt files in each source corpus directory,
shuffling, splitting and saving into multiple smaller files (50MB by default) in a target directory.

This script assumes the following source directory structures:
    - {src_dir}/{language}/{corpus}/*.txt
It will read from all source .txt files and create the following target directory structures:
    - {tgt_dir}/{language}/{corpus}
and within each target directory, it will create the following files:
    - train/*.txt
    - dev.txt
    - test.txt
Args:
    - src_root: root directory of the source.
    - tgt_root: root directory of the target.
    - langs: a list of language codes to process; if specified, languages not in this list will be ignored.
Note: edit the {EXCLUDED_FOLDERS} variable to exclude more folders in the source directory.
"""

import argparse
import glob
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tqdm import tqdm

EXCLUDED_FOLDERS = ['raw_corpus']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", default="src", help="Root directory with all source files.  Expected structure is root dir -> language dirs -> package dirs -> text files to process")
    parser.add_argument("tgt_root", default="tgt", help="Root directory with all target files.")
    parser.add_argument("--langs", default="", help="A list of language codes to process.  If not set, all languages under src_root will be processed.")
    parser.add_argument("--packages", default="", help="A list of packages to process.  If not set, all packages under the languages found will be processed.")
    parser.add_argument("--no_xz_output", default=True, dest="xz_output", action="store_false", help="Output compressed xz files")
    args = parser.parse_args()

    print("Processing files:")
    print(f"source root: {args.src_root}")
    print(f"target root: {args.tgt_root}")
    print("")

    langs = []
    if len(args.langs) > 0:
        langs = args.langs.split(',')
        print("Only processing the following languages: " + str(langs))

    packages = []
    if len(args.packages) > 0:
        packages = args.packages.split(',')
        print("Only processing the following packages: " + str(packages))

    src_root = Path(args.src_root)
    tgt_root = Path(args.tgt_root)

    lang_dirs = os.listdir(src_root)
    lang_dirs = [l for l in lang_dirs if l not in EXCLUDED_FOLDERS]    # skip excluded
    lang_dirs = [l for l in lang_dirs if os.path.isdir(src_root / l)]  # skip non-directory
    if len(langs) > 0: # filter languages if specified
        lang_dirs = [l for l in lang_dirs if l in langs]
    print(f"{len(lang_dirs)} total languages found:")
    print(lang_dirs)
    print("")

    for lang in lang_dirs:
        lang_root = src_root / lang
        data_dirs = os.listdir(lang_root)
        if len(packages) > 0:
            data_dirs = [d for d in data_dirs if d in packages]
        data_dirs = [d for d in data_dirs if os.path.isdir(lang_root / d)]
        print(f"{len(data_dirs)} total corpus found for language {lang}.")
        print(data_dirs)
        print("")

        for dataset_name in data_dirs:
            src_dir = lang_root / dataset_name
            tgt_dir = tgt_root / lang / dataset_name

            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            print(f"-> Processing {lang}-{dataset_name}")
            prepare_lm_data(src_dir, tgt_dir, lang, dataset_name, args.xz_output)

        print("")

def prepare_lm_data(src_dir, tgt_dir, lang, dataset_name, compress):
    """
    Combine, shuffle and split data into smaller files, following a naming convention.
    """
    assert isinstance(src_dir, Path)
    assert isinstance(tgt_dir, Path)
    with tempfile.TemporaryDirectory(dir=tgt_dir) as tempdir:
        tgt_tmp = os.path.join(tempdir, f"{lang}-{dataset_name}.tmp")
        print(f"--> Copying files into {tgt_tmp}...")
        # TODO: we can do this without the shell commands
        input_files = glob.glob(str(src_dir) + '/*.txt') + glob.glob(str(src_dir) + '/*.txt.xz') + glob.glob(str(src_dir) + '/*.txt.gz')
        for src_fn in tqdm(input_files):
            if src_fn.endswith(".txt"):
                cmd = f"cat {src_fn} >> {tgt_tmp}"
                subprocess.run(cmd, shell=True)
            elif src_fn.endswith(".txt.xz"):
                cmd = f"xzcat {src_fn} >> {tgt_tmp}"
                subprocess.run(cmd, shell=True)
            elif src_fn.endswith(".txt.gz"):
                cmd = f"zcat {src_fn} >> {tgt_tmp}"
                subprocess.run(cmd, shell=True)
            else:
                raise AssertionError("should not have found %s" % src_fn)
        tgt_tmp_shuffled = os.path.join(tempdir, f"{lang}-{dataset_name}.tmp.shuffled")

        print(f"--> Shuffling files into {tgt_tmp_shuffled}...")
        cmd = f"cat {tgt_tmp} | shuf > {tgt_tmp_shuffled}"
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to shuffle files!")
        size = os.path.getsize(tgt_tmp_shuffled) / 1024 / 1024 / 1024
        print(f"--> Shuffled file size: {size:.4f} GB")
        if size < 0.1:
            raise RuntimeError("Not enough data found to build a charlm.  At least 100MB data expected")

        print(f"--> Splitting into smaller files...")
        train_dir = tgt_dir / 'train'
        if not os.path.exists(train_dir): # make training dir
            os.makedirs(train_dir)
        cmd = f"split -C 52428800 -a 3 -d --additional-suffix .txt {tgt_tmp_shuffled} {train_dir}/{lang}-{dataset_name}-"
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to split files!")
        total = len(glob.glob(f'{train_dir}/*.txt'))
        print(f"--> {total} total files generated.")
        if total < 3:
            raise RuntimeError("Something went wrong!  %d file(s) produced by shuffle and split, expected at least 3" % total)

        print("--> Creating dev and test files...")
        dev_file = f"{tgt_dir}/dev.txt"
        test_file = f"{tgt_dir}/test.txt"
        shutil.move(f"{train_dir}/{lang}-{dataset_name}-000.txt", dev_file)
        shutil.move(f"{train_dir}/{lang}-{dataset_name}-001.txt", test_file)

        if compress:
            print("--> Compressing files...")
            txt_files = [dev_file, test_file] + glob.glob(f'{train_dir}/*.txt')
            for txt_file in tqdm(txt_files):
                subprocess.run(['xz', txt_file])

        print("--> Cleaning up...")
    print(f"--> All done for {lang}-{dataset_name}.\n")

if __name__ == "__main__":
    main()
