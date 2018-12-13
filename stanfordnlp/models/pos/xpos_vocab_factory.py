# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from stanfordnlp.models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(data, shorthand):
    if shorthand in ["af_afribooms", "grc_perseus", "ar_padt", "bg_btb", "cs_cac", "cs_fictree", "cs_pdt", "gl_ctg", "gl_treegal", "it_isdt", "it_postwita", "la_perseus", "lv_lvtb", "ro_rrt", "sk_snk", "sl_ssj", "sl_sst", "uk_iu"]:
        return XPOSVocab(data, shorthand, idx=2, sep="")
    elif shorthand in ["grc_proiel", "hy_armtdp", "eu_bdt", "br_keb", "bxr_bdt", "ca_ancora", "zh_gsd", "hr_set", "cs_pud", "da_ddt", "en_ewt", "en_gum", "en_pud", "et_edt", "fo_oft", "fi_pud", "fi_tdt", "fr_gsd", "fr_sequoia", "fr_spoken", "de_gsd", "got_proiel", "el_gdt", "he_htb", "hi_hdtb", "hu_szeged", "ga_idt", "ja_gsd", "ja_modern", "kk_ktb", "kmr_mg", "la_proiel", "pcm_nsc", "sme_giella", "no_bokmaal", "no_nynorsk", "no_nynorsklia", "cu_proiel", "fro_srcmf", "fa_seraji", "pt_bosque", "ru_syntagrus", "ru_taiga", "sr_set", "es_ancora", "sv_pud", "th_pud", "tr_imst", "hsb_ufal", "ug_udt", "vi_vtb"]:
        return WordVocab(data, shorthand, idx=2, ignore=["_"])
    elif shorthand in ["nl_alpino", "nl_lassysmall", "la_ittb", "sv_talbanken"]:
        return XPOSVocab(data, shorthand, idx=2, sep="|")
    elif shorthand in ["en_lines", "sv_lines", "ur_udtb"]:
        return XPOSVocab(data, shorthand, idx=2, sep="-")
    elif shorthand in ["fi_ftb"]:
        return XPOSVocab(data, shorthand, idx=2, sep=",")
    elif shorthand in ["id_gsd", "ko_gsd", "ko_kaist"]:
        return XPOSVocab(data, shorthand, idx=2, sep="+")
    elif shorthand in ["pl_lfg", "pl_sz"]:
        return XPOSVocab(data, shorthand, idx=2, sep=":")
    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))
