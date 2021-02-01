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

import subprocess
import glob
import os
import shutil
from pathlib import Path
import argparse

EXCLUDED_FOLDERS = ['raw_corpus']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_root", default="src", help="Root directory with all source files.")
    parser.add_argument("tgt_root", default="tgt", help="Root directory with all target files.")
    parser.add_argument("--langs", default="", help="A list of language codes to process.")
    args = parser.parse_args()

    print("Processing files:")
    print(f"source root: {args.src_root}")
    print(f"target root: {args.tgt_root}")
    print("")

    langs = []
    if len(args.langs) > 0:
        langs = args.langs.split(',')
        print("Only processing the following languages: " + str(langs))

    src_root = Path(args.src_root)
    tgt_root = Path(args.tgt_root)

    lang_dirs = os.listdir(src_root)
    lang_dirs = [l for l in lang_dirs if l not in EXCLUDED_FOLDERS] # skip excluded
    if len(langs) > 0: # filter languages if specified
        lang_dirs = [l for l in lang_dirs if l in langs]
    print(f"{len(lang_dirs)} total languages found:")
    print(lang_dirs)
    print("")

    for lang in lang_dirs:
        lang_root = src_root / lang
        data_dirs = os.listdir(lang_root)
        print(f"{len(data_dirs)} total corpus found for language {lang}.")

        for dataset_name in data_dirs:
            src_dir = lang_root / dataset_name
            tgt_dir = tgt_root / lang / dataset_name

            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            print(f"-> Processing {lang}-{dataset_name}")
            prepare_lm_data(src_dir, tgt_dir, lang, dataset_name)
        
        print("")

def prepare_lm_data(src_dir, tgt_dir, lang, dataset_name):
    """
    Combine, shuffle and split data into smaller files, following a naming convention.
    """
    assert isinstance(src_dir, Path)
    assert isinstance(tgt_dir, Path)
    tgt_tmp = tgt_dir / f"{lang}-{dataset_name}.tmp"
    if os.path.exists(tgt_tmp):
        os.remove(tgt_tmp)
    print(f"--> Copying files into {tgt_tmp}...")
    # TODO: we can do this without the shell commands
    for src_fn in glob.glob(str(src_dir) + '/*.txt'):
        cmd = f"cat {src_fn} >> {tgt_tmp}"
        subprocess.run(cmd, shell=True)
    for src_fn in glob.glob(str(src_dir) + '/*.txt.xz'):
        cmd = f"xzcat {src_fn} >> {tgt_tmp}"
        subprocess.run(cmd, shell=True)
    tgt_tmp_shuffled = Path(str(tgt_tmp) + ".shuffled")

    print(f"--> Shuffling files into {tgt_tmp_shuffled}...")
    cmd = f"cat {tgt_tmp} | shuf > {tgt_tmp_shuffled}"
    subprocess.run(cmd, shell=True)
    size = os.path.getsize(tgt_tmp_shuffled) / 1024 / 1024 / 1024
    print(f"--> Shuffled file size: {size:.4f} GB")

    print(f"--> Splitting into smaller files...")
    train_dir = tgt_dir / 'train'
    if not os.path.exists(train_dir): # make training dir
        os.makedirs(train_dir)
    cmd = f"split -C 52428800 -a 3 -d --additional-suffix .txt {tgt_tmp_shuffled} {train_dir}/{lang}-{dataset_name}-"
    subprocess.run(cmd, shell=True)
    total = len(glob.glob(f'{train_dir}/*.txt'))
    print(f"--> {total} total files generated.")

    print("--> Creating dev and test files...")
    shutil.move(f"{train_dir}/{lang}-{dataset_name}-000.txt", f"{tgt_dir}/dev.txt")
    shutil.move(f"{train_dir}/{lang}-{dataset_name}-001.txt", f"{tgt_dir}/test.txt")

    print("--> Cleaning up...")
    os.remove(tgt_tmp)
    os.remove(tgt_tmp_shuffled)
    print(f"--> All done for {lang}-{dataset_name}.\n")

if __name__ == "__main__":
    main()
