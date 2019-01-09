import os
import re
import sys

name_map = {'af_afribooms': 'UD_Afrikaans-AfriBooms', 'grc_perseus': 'UD_Ancient_Greek-Perseus', 'grc_proiel': 'UD_Ancient_Greek-PROIEL', 'ar_padt': 'UD_Arabic-PADT', 'hy_armtdp': 'UD_Armenian-ArmTDP', 'eu_bdt': 'UD_Basque-BDT', 'br_keb': 'UD_Breton-KEB', 'bg_btb': 'UD_Bulgarian-BTB', 'bxr_bdt': 'UD_Buryat-BDT', 'bxr_bdt_xv': 'UD_Buryat-BDT_XV', 'ca_ancora': 'UD_Catalan-AnCora', 'zh_gsd': 'UD_Chinese-GSD', 'hr_set': 'UD_Croatian-SET', 'cs_cac': 'UD_Czech-CAC', 'cs_fictree': 'UD_Czech-FicTree', 'cs_pdt': 'UD_Czech-PDT', 'cs_pud': 'UD_Czech-PUD', 'da_ddt': 'UD_Danish-DDT', 'nl_alpino': 'UD_Dutch-Alpino', 'nl_lassysmall': 'UD_Dutch-LassySmall', 'en_ewt': 'UD_English-EWT', 'en_gum': 'UD_English-GUM', 'en_lines': 'UD_English-LinES', 'en_pud': 'UD_English-PUD', 'et_edt': 'UD_Estonian-EDT', 'fo_oft': 'UD_Faroese-OFT', 'fi_ftb': 'UD_Finnish-FTB', 'fi_pud': 'UD_Finnish-PUD', 'fi_tdt': 'UD_Finnish-TDT', 'fr_gsd': 'UD_French-GSD', 'fr_sequoia': 'UD_French-Sequoia', 'fr_spoken': 'UD_French-Spoken', 'gl_ctg': 'UD_Galician-CTG', 'gl_treegal': 'UD_Galician-TreeGal', 'de_gsd': 'UD_German-GSD', 'got_proiel': 'UD_Gothic-PROIEL', 'el_gdt': 'UD_Greek-GDT', 'he_htb': 'UD_Hebrew-HTB', 'hi_hdtb': 'UD_Hindi-HDTB', 'hu_szeged': 'UD_Hungarian-Szeged', 'id_gsd': 'UD_Indonesian-GSD', 'ga_idt': 'UD_Irish-IDT', 'ga_idt_xv': 'UD_Irish-IDT_XV', 'it_isdt': 'UD_Italian-ISDT', 'it_postwita': 'UD_Italian-PoSTWITA', 'ja_gsd': 'UD_Japanese-GSD', 'ja_modern': 'UD_Japanese-Modern', 'kk_ktb': 'UD_Kazakh-KTB', 'ko_gsd': 'UD_Korean-GSD', 'ko_kaist': 'UD_Korean-Kaist', 'kmr_mg': 'UD_Kurmanji-MG', 'kmr_mg_xv': 'UD_Kurmanji-MG_XV', 'la_ittb': 'UD_Latin-ITTB', 'la_perseus': 'UD_Latin-Perseus', 'la_proiel': 'UD_Latin-PROIEL', 'lv_lvtb': 'UD_Latvian-LVTB', 'pcm_nsc': 'UD_Naija-NSC', 'sme_giella': 'UD_North_Sami-Giella', 'no_bokmaal': 'UD_Norwegian-Bokmaal', 'no_nynorsk': 'UD_Norwegian-Nynorsk', 'no_nynorsklia': 'UD_Norwegian-NynorskLIA', 'cu_proiel': 'UD_Old_Church_Slavonic-PROIEL', 'fro_srcmf': 'UD_Old_French-SRCMF', 'fa_seraji': 'UD_Persian-Seraji', 'pl_lfg': 'UD_Polish-LFG', 'pl_sz': 'UD_Polish-SZ', 'pt_bosque': 'UD_Portuguese-Bosque', 'ro_rrt': 'UD_Romanian-RRT', 'ru_syntagrus': 'UD_Russian-SynTagRus', 'ru_taiga': 'UD_Russian-Taiga', 'sr_set': 'UD_Serbian-SET', 'sk_snk': 'UD_Slovak-SNK', 'sl_ssj': 'UD_Slovenian-SSJ', 'sl_sst': 'UD_Slovenian-SST', 'es_ancora': 'UD_Spanish-AnCora', 'sv_lines': 'UD_Swedish-LinES', 'sv_pud': 'UD_Swedish-PUD', 'sv_talbanken': 'UD_Swedish-Talbanken', 'th_pud': 'UD_Thai-PUD', 'tr_imst': 'UD_Turkish-IMST', 'uk_iu': 'UD_Ukrainian-IU', 'hsb_ufal': 'UD_Upper_Sorbian-UFAL', 'hsb_ufal_xv': 'UD_Upper_Sorbian-UFAL_XV', 'ur_udtb': 'UD_Urdu-UDTB', 'ug_udt': 'UD_Uyghur-UDT', 'vi_vtb': 'UD_Vietnamese-VTB'}

# get list of report files
report_files_dir = sys.argv[1]
report_files = os.listdir(report_files_dir)
report_files.sort()

f1_header_regex = re.compile('F1 Score')

for report_file in report_files:
    contents = open(report_files_dir+'/'+report_file).read().split('\n')[:-1]
    row = name_map[report_file.split('.')[0]]
    for line in contents:
        if len(line.split('|')) > 3 and not f1_header_regex.search(line):
            row += (','+line.split('|')[3].rstrip().lstrip())
    print(row)
