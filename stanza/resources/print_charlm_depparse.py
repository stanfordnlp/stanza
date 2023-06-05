"""
A small utility script to output which depparse models use charlm

(It should skip en_genia, en_craft, but currently doesn't)

Not frequently useful, but seems like the kind of thing that might get used a couple times
"""

from stanza.resources.common import load_resources_json
from stanza.resources.default_packages import default_charlms, depparse_charlms

def list_depparse():
    charlm_langs = list(default_charlms.keys())
    resources = load_resources_json()

    models = ["%s_%s" % (lang, model) for lang in charlm_langs for model in resources[lang].get("depparse", {})
              if lang not in depparse_charlms or model not in depparse_charlms[lang] or depparse_charlms[lang][model] is not None]
    return models

if __name__ == "__main__":
    models = list_depparse()
    print(" ".join(models))
