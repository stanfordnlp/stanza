"""
Kazakh Transliteration:
    Cyrillic Kazakh --> Latin Kazakh


"""

import argparse
import os
from re import M
import string
import sys

from stanza.models.common.utils import open_read_text, get_tqdm
tqdm = get_tqdm()

"""
This dictionary isn't used in the code, just put this here in case you want to implement it more
efficiently and in case the need to look up the unicode encodings for these letters might arise.
Some letters are mapped to multiple latin letters, for these, I separated the unicde with a '%' delimiter
between the two unicode characters.
"""
alph_map = {
    '\u0410' # А
    : '\u0041', # A
    '\u0430' # а
    : '\u0061', # a

    '\u04D8' # Ә
    : '\u00c4', # Ä
    '\u04D9' # ә
    : '\u00e4', # ä

    '\u0411' # Б
    : '\u0042', # B
    '\u0431' # б
    : '\u0062', # b

    '\u0412' # В
    : '\u0056', # V
    '\u0432' # в
    : '\u0076', # v

    '\u0413' # Г
    : '\u0047', # G
    '\u0433' # г
    : '\u0067', # g

    '\u0492' # Ғ
    : '\u011e', # Ğ
    '\u0493' # ғ
    : '\u011f', # ğ

    '\u0414' # Д
    : '\u0044', # D
    '\u0434' # д
    : '\u0064', # d

    '\u0415' # Е
    : '\u0045', # E
    '\u0435' # е
    : '\u0065', # e

    '\u0401' # Ё
    : '\u0130%\u006f', # İo
    '\u0451' # ё
    : '\u0069%\u006f', #io

    '\u0416' # Ж
    : '\u004a', # J
    '\u0436' # ж
    : '\u006a', # j

    '\u0417' # З
    : '\u005a', # Z
    '\u0437' # з
    : '\u007a', # z

    '\u0418' # И
    : '\u0130', # İ
    '\u0438' # и
    : '\u0069', # i

    '\u0419' # Й
    : '\u0130', # İ
    '\u0439' # й
    : '\u0069', # i

    '\u041A' # К
    : '\u004b', # K
    '\u043A' # к
    : '\u006b', # k

    '\u049A' # Қ
    : '\u0051', # Q
    '\u049B' # қ
    : '\u0071', # q

    '\u041B' # Л
    : '\u004c', # L
    '\u043B' # л
    : '\u006c', # l

    '\u041C' # М
    : '\u004d', # M
    '\u043C' # м
    : '\u006d', # m

    '\u041D' # Н
    : '\u004e', # N
    '\u043D' # н
    : '\u006e', # n

    '\u04A2' # Ң
    : '\u00d1', # Ñ
    '\u04A3' # ң
    : '\u00f1', # ñ

    '\u041E' # О
    : '\u004f', # O
    '\u043E' # о
    : '\u006f', # o

    '\u04E8' # Ө
    : '\u00d6', # Ö
    '\u04E9' # ө
    : '\u00f6', # ö

    '\u041F' # П
    : '\u0050', # P
    '\u043F' # п
    : '\u0070', # p

    '\u0420' # Р
    : '\u0052', # R
    '\u0440' # р
    : '\u0072', # r

    '\u0421' # С
    : '\u0053', # S
    '\u0441' # с
    : '\u0073', # s

    '\u0422' # Т
    : '\u0054', # T
    '\u0442' # т
    : '\u0074', # t

    '\u0423' # У
    : '\u0055', # U
    '\u0443' # у
    : '\u0075', # u

    '\u04B0' # Ұ
    : '\u016a', # Ū
    '\u04B1' # ұ
    : '\u016b', # ū

    '\u04AE' # Ү
    : '\u00dc', # Ü
    '\u04AF' # ү
    : '\u00fc', # ü

    '\u0424' # Ф
    : '\u0046', # F
    '\u0444' # ф
    : '\u0066', # f

    '\u0425' # Х
    : '\u0048', # H
    '\u0445' # х
    : '\u0068', # h

    '\u04BA' # Һ
    : '\u0048', # H
    '\u04BB' # һ
    : '\u0068', # h

    '\u0426' # Ц
    : '\u0043', # C
    '\u0446' # ц
    : '\u0063', # c

    '\u0427' # Ч
    : '\u00c7', # Ç
    '\u0447' # ч
    : '\u00e7', # ç

    '\u0428' # Ш
    : '\u015e', # Ş
    '\u0448' # ш
    : '\u015f', # ş

    '\u0429' # Щ
    : '\u015e%\u00e7', # Şç
    '\u0449' # щ
    : '\u015f%\u00e7', # şç

    '\u042A' # Ъ
    : '', # Empty String
    '\u044A' # ъ
    : '', # Empty String \u

    '\u042B' # Ы
    : '\u0059', # Y
    '\u044B' # ы
    : '\u0079', # y

    '\u0406' # І
    : '\u0130', # İ
    '\u0456' # і
    : '\u0069', # i

    '\u042C' # Ь
    : '', # Empty String
    '\u044C' # ь
    : '', # Empty String

    '\u042D' # Э
    : '\u0045', # E
    '\u044D' # э
    : '\u0065', # e

    '\u042E' # Ю
    : '\u0130%\u0075', # İu
    '\u044E' # ю
    : '\u0069%\u0075', # iu

    '\u042F' # Я
    : '\u0130%\u0061', # İa
    '\u044F' # я
    : '\u0069%\u0061' # ia
}

kazakh_alph = "АаӘәБбВвГгҒғДдЕеЁёЖжЗзИиЙйКкҚқЛлМмНнҢңОоӨөПпРрСсТтУуҰұҮүФфХхҺһЦцЧчШшЩщЪъЫыІіЬьЭэЮюЯя"
latin_alph = "AaÄäBbVvGgĞğDdEeİoioJjZzİiİiKkQqLlMmNnÑñOoÖöPpRrSsTtUuŪūÜüFfHhHhCcÇçŞşŞçşçYyİiEeİuiuİaia"
mult_mapping = "ЁёЩщЮюЯя"
empty_mapping = "ЪъЬь"


"""
ϵ : Ukrainian letter for 'ё'
ə : Russian utf-8 encoding for Kazakh 'ә'
ó : 2016 Kazakh Latin adopted this instead of 'ö'
ã : 1 occurrence in the dataset -- mapped to 'a'

"""
russian_alph = "ϵəóã"
russian_counterpart = "ioäaöa"
def create_dic(source_alph, target_alph, mult_mapping, empty_mapping):
    res = {}
    idx = 0
    for i in range(len(source_alph)):
        l_s = source_alph[i]
        if l_s in mult_mapping:
            res[l_s] = target_alph[idx] + target_alph[idx+1]
            idx += 1
        elif l_s in empty_mapping:
            res[l_s] = ''
            idx -= 1
        else:
            res[l_s] = target_alph[idx]
        idx += 1

    res['ϵ'] = 'io'
    res['ə'] = 'ä'
    res['ó'] = 'ö'
    res['ã'] = 'a'

    print(res)
    return res


supp_alph = "IWwXx0123456789–«»—"

def transliterate(source):
    output = ""
    tr_dict = create_dic(kazakh_alph, latin_alph, mult_mapping, empty_mapping)
    punc = string.punctuation
    white_spc = string.whitespace
    for c in source:
        if c in punc or c in white_spc:
            output += c

        elif c in latin_alph or c in supp_alph:
            output += c

        elif c in tr_dict:
            output += tr_dict[c]

        else:
            print(f"Transliteration Error: {c}")

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, nargs="+", help="Files to process")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to output results")
    args = parser.parse_args()

    tr_dict = create_dic(kazakh_alph, latin_alph, mult_mapping, empty_mapping)
    for filename in tqdm(args.input_file):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            directory, basename = os.path.split(filename)
            output_name = os.path.join(args.output_dir, basename)
            if output_name.endswith(".xz"):
                output_name = output_name[:-3]
            output_name = output_name + ".trans"
        else:
            output_name = filename + ".trans"

        tqdm.write("Transliterating %s to %s" % (filename, output_name))

        with open_read_text(filename) as f_in:
            data = f_in.read()
        with open(output_name, 'w') as f_out:
            punc = string.punctuation
            white_spc = string.whitespace
            for c in tqdm(data, leave=False):
                if c in tr_dict:
                    f_out.write(tr_dict[c])

                else:
                    f_out.write(c)


    print("Process Completed Successfully!")

