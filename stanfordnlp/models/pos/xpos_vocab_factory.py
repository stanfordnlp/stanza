# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from stanfordnlp.models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(data, shorthand):
    if shorthand in ["af_afribooms", "grc_perseus", "ar_padt", "bg_btb", "hr_set", "cs_cac", "cs_cltt", "cs_fictree", "cs_pdt", "en_partut", "fr_partut", "gl_ctg", "it_isdt", "it_partut", "it_postwita", "it_twittiro", "it_vit", "ja_gsd", "lv_lvtb", "lt_alksnis", "ro_nonstandard", "ro_rrt", "gd_arcosg", "sr_set", "sk_snk", "sl_ssj", "ta_ttb", "uk_iu", "gl_treegal", "la_perseus", "sl_sst"]:
        return XPOSVocab(data, shorthand, idx=2, sep="")
    elif shorthand in ["grc_proiel", "hy_armtdp", "eu_bdt", "be_hse", "ca_ancora", "zh_gsd", "zhs_gsdsimp", "lzh_kyoto", "cop_scriptorium", "da_ddt", "en_ewt", "en_gum", "et_edt", "fi_tdt", "fr_ftb", "fr_gsd", "fr_sequoia", "fr_spoken", "de_gsd", "de_hdt", "got_proiel", "el_gdt", "he_htb", "hi_hdtb", "hu_szeged", "ga_idt", "ja_bccwj", "la_proiel", "lt_hse", "mt_mudt", "mr_ufal", "no_bokmaal", "no_nynorsk", "no_nynorsklia", "cu_proiel", "fro_srcmf", "orv_torot", "fa_seraji", "pt_bosque", "pt_gsd", "ru_gsd", "ru_syntagrus", "ru_taiga", "es_ancora", "es_gsd", "swl_sslc", "te_mtg", "tr_imst", "ug_udt", "vi_vtb", "wo_wtb", "bxr_bdt", "et_ewt", "kk_ktb", "kmr_mg", "olo_kkpp", "sme_giella", "hsb_ufal"]:
        return WordVocab(data, shorthand, idx=2, ignore=["_"])
    elif shorthand in ["nl_alpino", "nl_lassysmall", "la_ittb", "sv_talbanken"]:
        return XPOSVocab(data, shorthand, idx=2, sep="|")
    elif shorthand in ["en_lines", "sv_lines", "ur_udtb"]:
        return XPOSVocab(data, shorthand, idx=2, sep="-")
    elif shorthand in ["fi_ftb"]:
        return XPOSVocab(data, shorthand, idx=2, sep=",")
    elif shorthand in ["id_gsd", "ko_gsd", "ko_kaist"]:
        return XPOSVocab(data, shorthand, idx=2, sep="+")
    elif shorthand in ["pl_lfg", "pl_pdb"]:
        return XPOSVocab(data, shorthand, idx=2, sep=":")
    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))
