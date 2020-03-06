import sys

backoff_models = { "UD_Breton-KEB": "ga_idt",
                   "UD_Czech-PUD": "cs_pdt",
                   "UD_English-PUD": "en_ewt",
                   "UD_Faroese-OFT": "nn_nynorsk",
                   "UD_Finnish-PUD": "fi_tdt",
                   "UD_Japanese-Modern": "ja_gsd",
                   "UD_Naija-NSC": "en_ewt",
                   "UD_Swedish-PUD": "sv_talbanken"
                 }

print(backoff_models[sys.argv[1]])
