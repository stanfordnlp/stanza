# UD_Bhojpuri-BHTB XPOS -> BIS
XPOS_TO_BIS = {
    "N_NN":    "NN",
    "N_NNC":   "NN",
    "N_NNP":   "NNP",
    "N_NST":   "NST",

    "V_VM":    "VM",
    "V_VMP":   "VM",
    "V_VAUX":  "VAUX",

    "PR_PRP":  "PRP",
    "PR_PRF":  "PRP",
    "PR_PRI":  "PRP",
    "PR_PRL":  "PRP",
    "PR_PRQ":  "WQ",

    "DM_DMD":  "DEM",
    "DM_DMR":  "DEM",
    "DM_DMI":  "DEM",
    "DM_DMQ":  "WQ",

    "CC_CCD":  "CC",
    "CC_CCS":  "CC",

    # IIT has no QT category, so these BIS tags do not appear in that data
    "QT_QTC":  "QC",
    "QT_QTF":  "QF",
    "QT_QTO":  "QO",

    "RP_RPD":  "RP",
    "RP_NEG":  "NEG",
    "RP_INTF": "INTF",
    "RP_INJ":  "INJ",
    "RP_CL":   "CL",

    "RD_PUNC": "SYM",
    "RD_SYM":  "SYM",
    "RD_ECH":  "ECH",

    "PSP":     "PSP",
    "JJ":      "JJ",
    "RB":      "RB",
}
