# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(vocabfile, data, shorthand):
    if shorthand in ["af_afribooms", "grc_proiel", "hy_armtdp", "eu_bdt", "br_keb", "bxr_bdt", "ca_ancora", "zh_gsd", "hr_set", "cs_pud", "da_ddt", "nl_alpino", "nl_lassysmall", "en_ewt", "en_gum", "en_lines", "en_pud", "et_edt", "fo_oft", "fi_pud", "fi_tdt", "fr_gsd", "fr_sequoia", "fr_spoken", "gl_ctg", "de_gsd", "got_proiel", "el_gdt", "he_htb", "hi_hdtb", "hu_szeged", "id_gsd", "ga_idt", "it_isdt", "it_postwita", "ja_gsd", "ja_modern", "kk_ktb", "kmr_mg", "la_proiel", "pcm_nsc", "sme_giella", "no_bokmaal", "no_nynorsk", "no_nynorsklia", "cu_proiel", "fro_srcmf", "fa_seraji", "pt_bosque", "ru_syntagrus", "ru_taiga", "sr_set", "es_ancora", "sv_lines", "sv_pud", "sv_talbanken", "th_pud", "tr_imst", "hsb_ufal", "ur_udtb", "ug_udt", "vi_vtb"]:
        return WordVocab(vocabfile, data, shorthand, idx=2)
    elif shorthand in ["grc_perseus", "ar_padt", "bg_btb", "cs_cac", "cs_fictree", "cs_pdt", "fi_ftb", "gl_treegal", "la_perseus", "lv_lvtb", "pl_lfg", "pl_sz", "ro_rrt", "sk_snk", "sl_ssj", "sl_sst", "uk_iu"]:
        return XPOSVocab(vocabfile, data, shorthand, idx=2, sep="")
    elif shorthand in ["ko_gsd", "ko_kaist"]:
        return XPOSVocab(vocabfile, data, shorthand, idx=2, sep="+")
    elif shorthand in ["la_ittb"]:
        return XPOSVocab(vocabfile, data, shorthand, idx=2, sep="|")
    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))
