"""
Script to renormalize diacritics for Vietnamese text

from BARTpho
https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md
https://github.com/VinAIResearch/BARTpho/blob/main/LICENSE

MIT License

Copyright (c) 2021 VinAI Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os

DICT_MAP = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}


def replace_all(text):
    for i, j in DICT_MAP.items():
        text = text.replace(i, j)
    return text

def convert_file(org_file, new_file):
    with open(org_file, 'r', encoding='utf-8') as reader, open(new_file, 'w', encoding='utf-8') as writer:
        content = reader.readlines()
        for line in content:
            new_line = replace_all(line)
            writer.write(new_line)

def convert_files(file_list, new_dir):
    for file_name in file_list:
        base_name = os.path.split(file_name)[-1]
        new_file_path = os.path.join(new_dir, base_name)

        convert_file(file_name, new_file_path)


def convert_dir(org_dir, new_dir, suffix):
    os.makedirs(new_dir, exist_ok=True)
    file_list = os.listdir(org_dir)
    file_list = [os.path.join(org_dir, f) for f in file_list if os.path.splitext(f)[1] == suffix]
    convert_files(file_list, new_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Script that renormalizes diacritics'
    )

    parser.add_argument(
        'orig',
        help='Location of the original directory'
    )

    parser.add_argument(
        'converted',
        help='The location of new directory'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='.txt',
        help='Which suffix to look for when renormalizing a directory'
    )

    args = parser.parse_args()

    if os.path.isfile(args.orig):
        convert_file(args.orig, args.converted)
    else:
        convert_dir(args.orig, args.converted, args.suffix)


if __name__ == '__main__':
    main()
