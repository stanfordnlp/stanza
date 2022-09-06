"""Converts raw data files into json files usable by the training script.

Currently it supports converting wikiner datasets, available here:
  https://figshare.com/articles/dataset/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500
  - download the language of interest to {Language}-WikiNER
  - then run
    prepare_ner_dataset.py French-WikiNER

Also, Finnish Turku dataset, available here:
  - https://turkunlp.org/fin-ner.html
  - https://github.com/TurkuNLP/turku-ner-corpus
    git clone the repo into $NERBASE/finnish
    you will now have a directory
    $NERBASE/finnish/turku-ner-corpus
  - prepare_ner_dataset.py fi_turku

FBK in Italy produced an Italian dataset.
  - KIND: an Italian Multi-Domain Dataset for Named Entity Recognition
    Paccosi T. and Palmero Aprosio A.
    LREC 2022
  - https://arxiv.org/abs/2112.15099
  The processing here is for a combined .tsv file they sent us.
  - prepare_ner_dataset.py it_fbk
  There is a newer version of the data available here:
    https://github.com/dhfbk/KIND
  TODO: update to the newer version of the data

IJCNLP 2008 produced a few Indian language NER datasets.
  description:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=3
  download:
    http://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=5
  The models produced from these datasets have extremely low recall, unfortunately.
  - prepare_ner_dataset.py hi_ijc

FIRE 2013 also produced NER datasets for Indian languages.
  http://au-kbc.org/nlp/NER-FIRE2013/index.html
  The datasets are password locked.
  For Stanford users, contact Chris Manning for license details.
  For external users, please contact the organizers for more information.
  - prepare_ner_dataset.py hi-fire2013

HiNER is another Hindi dataset option
  https://github.com/cfiltnlp/HiNER
  - HiNER: A Large Hindi Named Entity Recognition Dataset
    Murthy, Rudra and Bhattacharjee, Pallab and Sharnagat, Rahul and
    Khatri, Jyotsana and Kanojia, Diptesh and Bhattacharyya, Pushpak
  There are two versions:
    hi_hinercollapsed and hi_hiner
  The collapsed version has just PER, LOC, ORG
  - convert data as follows:
    cd $NERBASE
    mkdir hindi
    cd hindi
    git clone git@github.com:cfiltnlp/HiNER.git
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset hi_hiner
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset hi_hinercollapsed

Ukranian NER is provided by lang-uk, available here:
  https://github.com/lang-uk/ner-uk
  git clone the repo to $NERBASE/lang-uk
  There should be a subdirectory $NERBASE/lang-uk/ner-uk/data at that point
  Conversion script graciously provided by Andrii Garkavyi @gawy
  - prepare_ner_dataset.py uk_languk

There are two Hungarian datasets are available here:
  https://rgai.inf.u-szeged.hu/node/130
  http://www.lrec-conf.org/proceedings/lrec2006/pdf/365_pdf.pdf
  We combined them and give them the label hu_rgai
  You can also build individual pieces with hu_rgai_business or hu_rgai_criminal
  Create a subdirectory of $NERBASE, $NERBASE/hu_rgai, and download both of
    the pieces and unzip them in that directory.
  - prepare_ner_dataset.py hu_rgai

Another Hungarian dataset is here:
  - https://github.com/nytud/NYTK-NerKor
  - git clone the entire thing in your $NERBASE directory to operate on it
  - prepare_ner_dataset.py hu_nytk

The two Hungarian datasets can be combined with hu_combined
  TODO: verify that there is no overlap in text
  - prepare_ner_dataset.py hu_combined

BSNLP publishes NER datasets for Eastern European languages.
  - In 2019 they published BG, CS, PL, RU.
  - http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html
  - In 2021 they added some more data, but the test sets
    were not publicly available as of April 2021.
    Therefore, currently the model is made from 2019.
    In 2021, the link to the 2021 task is here:
    http://bsnlp.cs.helsinki.fi/shared-task.html
  - The below method processes the 2019 version of the corpus.
    It has specific adjustments for the BG section, which has
    quite a few typos or mis-annotations in it.  Other languages
    probably need similar work in order to function optimally.
  - make a directory $NERBASE/bsnlp2019
  - download the "training data are available HERE" and
    "test data are available HERE" to this subdirectory
  - unzip those files in that directory
  - we use the code name "bg_bsnlp19".  Other languages from
    bsnlp 2019 can be supported by adding the appropriate
    functionality in convert_bsnlp.py.
  - prepare_ner_dataset.py bg_bsnlp19

NCHLT produced NER datasets for many African languages.
  Unfortunately, it is difficult to make use of many of these,
  as there is no corresponding UD data from which to build a
  tokenizer or other tools.
  - Afrikaans:  https://repo.sadilar.org/handle/20.500.12185/299
  - isiNdebele: https://repo.sadilar.org/handle/20.500.12185/306
  - isiXhosa:   https://repo.sadilar.org/handle/20.500.12185/312
  - isiZulu:    https://repo.sadilar.org/handle/20.500.12185/319
  - Sepedi:     https://repo.sadilar.org/handle/20.500.12185/328
  - Sesotho:    https://repo.sadilar.org/handle/20.500.12185/334
  - Setswana:   https://repo.sadilar.org/handle/20.500.12185/341
  - Siswati:    https://repo.sadilar.org/handle/20.500.12185/346
  - Tsivenda:   https://repo.sadilar.org/handle/20.500.12185/355
  - Xitsonga:   https://repo.sadilar.org/handle/20.500.12185/362
  Agree to the license, download the zip, and unzip it in
  $NERBASE/NCHLT

UCSY built a Myanmar dataset.  They have not made it publicly
  available, but they did make it available to Stanford for research
  purposes.  Contact Chris Manning or John Bauer for the data files if
  you are Stanford affiliated.
  - https://arxiv.org/abs/1903.04739
  - Syllable-based Neural Named Entity Recognition for Myanmar Language
    by Hsu Myat Mo and Khin Mar Soe

Hanieh Poostchi et al produced a Persian NER dataset:
  - git@github.com:HaniehP/PersianNER.git
  - https://github.com/HaniehP/PersianNER
  - Hanieh Poostchi, Ehsan Zare Borzeshi, Mohammad Abdous, and Massimo Piccardi,
    "PersoNER: Persian Named-Entity Recognition"
  - Hanieh Poostchi, Ehsan Zare Borzeshi, and Massimo Piccardi,
    "BiLSTM-CRF for Persian Named-Entity Recognition; ArmanPersoNERCorpus: the First Entity-Annotated Persian Dataset"
  - Conveniently, this dataset is already in BIO format.  It does not have a dev split, though.
    git clone the above repo, unzip ArmanPersoNERCorpus.zip, and this script will split the
    first train fold into a dev section.

SUC3 is a Swedish NER dataset provided by Språkbanken
  - https://spraakbanken.gu.se/en/resources/suc3
  - The splitting tool is generously provided by
    Emil Stenstrom
    https://github.com/EmilStenstrom/suc_to_iob
  - Download the .bz2 file at this URL and put it in $NERBASE/sv_suc3shuffle
    It is not necessary to unzip it.
  - Gustafson-Capková, Sophia and Britt Hartmann, 2006, 
    Manual of the Stockholm Umeå Corpus version 2.0.
    Stockholm University.
  - Östling, Robert, 2013, Stagger 
    an Open-Source Part of Speech Tagger for Swedish
    Northern European Journal of Language Technology 3: 1–18
    DOI 10.3384/nejlt.2000-1533.1331
  - The shuffled dataset can be converted with dataset code
    prepare_ner_dataset.py sv_suc3shuffle
  - If you fill out the license form and get the official data,
    you can get the official splits by putting the provided zip file
    in $NERBASE/sv_suc3licensed.  Again, not necessary to unzip it
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset sv_suc3licensed

DDT is a reformulation of the Danish Dependency Treebank as an NER dataset
  - https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dane
  - direct download link as of late 2021: https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip
  - https://aclanthology.org/2020.lrec-1.565.pdf
    DaNE: A Named Entity Resource for Danish
    Rasmus Hvingelby, Amalie Brogaard Pauli, Maria Barrett,
    Christina Rosted, Lasse Malm Lidegaard, Anders Søgaard
  - place ddt.zip in $NERBASE/da_ddt/ddt.zip
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset da_ddt

NorNE is the Norwegian Dependency Treebank with NER labels
  - LREC 2020
    NorNE: Annotating Named Entities for Norwegian
    Fredrik Jørgensen, Tobias Aasmoe, Anne-Stine Ruud Husevåg,
    Lilja Øvrelid, and Erik Velldal
  - both Bokmål and Nynorsk
  - This dataset is in a git repo:
    https://github.com/ltgoslo/norne
    Clone it into $NERBASE
    git clone git@github.com:ltgoslo/norne.git
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset nb_norne
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset nn_norne

tr_starlang is a set of constituency trees for Turkish
  The words in this dataset (usually) have NER labels as well

  A dataset in three parts from the Starlang group in Turkey:
  Neslihan Kara, Büşra Marşan, et al
    Creating A Syntactically Felicitous Constituency Treebank For Turkish
    https://ieeexplore.ieee.org/document/9259873
  git clone the following three repos
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-15
    https://github.com/olcaytaner/TurkishAnnotatedTreeBank2-20
  Put them in
    $CONSTITUENCY_HOME/turkish    (yes, the constituency home)
  python3 -m stanza.utils.datasets.ner.prepare_ner_dataset tr_starlang

GermEval2014 is a German NER dataset
  https://sites.google.com/site/germeval2014ner/data
  https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J
  Download the files in that directory
    NER-de-train.tsv NER-de-dev.tsv NER-de-test.tsv
  put them in
    $NERBASE/germeval2014
  then run
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset de_germeval2014

The UD Japanese GSD dataset has a conversion by Megagon Labs
  https://github.com/megagonlabs/UD_Japanese-GSD
  https://github.com/megagonlabs/UD_Japanese-GSD/tags
  - r2.9-NE has the NE tagged files inside a "spacy"
    folder in the download
  - expected directory for this data:
    unzip the .zip of the release into
      $NERBASE/ja_gsd
    so it should wind up in
      $NERBASE/ja_gsd/UD_Japanese-GSD-r2.9-NE
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset ja_gsd

L3Cube is a Marathi dataset
  - https://arxiv.org/abs/2204.06029
    https://arxiv.org/pdf/2204.06029.pdf
    https://github.com/l3cube-pune/MarathiNLP
  - L3Cube-MahaNER: A Marathi Named Entity Recognition Dataset and BERT models
    Parth Patil, Aparna Ranade, Maithili Sabane, Onkar Litake, Raviraj Joshi

  Clone the repo into $NERBASE/marathi
    git clone git@github.com:l3cube-pune/MarathiNLP.git
  Then run
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset mr_l3cube

Daffodil University produced a Bangla NER dataset
  - https://github.com/Rifat1493/Bengali-NER
  - https://ieeexplore.ieee.org/document/8944804
  - Bengali Named Entity Recognition:
    A survey with deep learning benchmark
    Md Jamiur Rahman Rifat, Sheikh Abujar, Sheak Rashed Haider Noori,
    Syed Akhter Hossain

  Clone the repo into a "bangla" subdirectory of $NERBASE
    cd $NERBASE/bangla
    git clone git@github.com:Rifat1493/Bengali-NER.git
  Then run
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset bn_daffodil

LST20 is a Thai NER dataset from 2020
  - https://arxiv.org/abs/2008.05055
    The Annotation Guideline of LST20 Corpus
    Prachya Boonkwan, Vorapon Luantangsrisuk, Sitthaa Phaholphinyo,
    Kanyanat Kriengket, Dhanon Leenoi, Charun Phrombut,
    Monthika Boriboon, Krit Kosawat, Thepchai Supnithi
  - This script processes a version which can be downloaded here after registration:
    https://aiforthai.in.th/index.php
  - There is another version downloadable from HuggingFace
    The script will likely need some modification to be compatible
    with the HuggingFace version
  - Download the data in $NERBASE/thai/LST20_Corpus
    There should be "train", "eval", "test" directories after downloading
  - Then run
    pytohn3 -m stanza.utils.datasets.ner.prepare_ner_dataset th_lst20

NKJP is a Polish NER dataset
  - http://nkjp.pl/index.php?page=0&lang=1
    About the Project
  - http://zil.ipipan.waw.pl/DistrNKJP
    Wikipedia subcorpus used to train charlm model
  - http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=view&target=NKJP-PodkorpusMilionowy-1.2.tar.gz
    Annotated subcorpus to train NER model.
    Download and extract to $NERBASE/Polish-NKJP or leave the gzip in $NERBASE/polish/...

kk_kazNERD is a Kazakh dataset published in 2021
  - https://github.com/IS2AI/KazNERD
  - https://arxiv.org/abs/2111.13419
    KazNERD: Kazakh Named Entity Recognition Dataset
    Rustem Yeshpanov, Yerbolat Khassanov, Huseyin Atakan Varol
  - in $NERBASE, make a "kazakh" directory, then git clone the repo there
    mkdir -p $NERBASE/kazakh
    cd $NERBASE/kazakh
    git clone git@github.com:IS2AI/KazNERD.git
  - Then run
    pytohn3 -m stanza.utils.datasets.ner.prepare_ner_dataset kk_kazNERD

Masakhane NER is a set of NER datasets for African languages
  - MasakhaNER: Named Entity Recognition for African Languages
    Adelani, David Ifeoluwa; Abbott, Jade; Neubig, Graham;
    D’souza, Daniel; Kreutzer, Julia; Lignos, Constantine;
    Palen-Michel, Chester; Buzaaba, Happy; Rijhwani, Shruti;
    Ruder, Sebastian; Mayhew, Stephen; Azime, Israel Abebe;
    Muhammad, Shamsuddeen H.; Emezue, Chris Chinenye;
    Nakatumba-Nabende, Joyce; Ogayo, Perez; Anuoluwapo, Aremu;
    Gitau, Catherine; Mbaye, Derguene; Alabi, Jesujoba;
    Yimam, Seid Muhie; Gwadabe, Tajuddeen Rabiu; Ezeani, Ignatius;
    Niyongabo, Rubungo Andre; Mukiibi, Jonathan; Otiende, Verrah;
    Orife, Iroro; David, Davis; Ngom, Samba; Adewumi, Tosin;
    Rayson, Paul; Adeyemi, Mofetoluwa; Muriuki, Gerald;
    Anebi, Emmanuel; Chukwuneke, Chiamaka; Odu, Nkiruka;
    Wairagala, Eric Peter; Oyerinde, Samuel; Siro, Clemencia;
    Bateesa, Tobius Saul; Oloyede, Temilola; Wambui, Yvonne;
    Akinode, Victor; Nabagereka, Deborah; Katusiime, Maurice;
    Awokoya, Ayodele; MBOUP, Mouhamadane; Gebreyohannes, Dibora;
    Tilaye, Henok; Nwaike, Kelechi; Wolde, Degaga; Faye, Abdoulaye;
    Sibanda, Blessing; Ahia, Orevaoghene; Dossou, Bonaventure F. P.;
    Ogueji, Kelechi; DIOP, Thierno Ibrahima; Diallo, Abdoulaye;
    Akinfaderin, Adewale; Marengereke, Tendai; Osei, Salomey
  - https://github.com/masakhane-io/masakhane-ner
  - git clone the repo to $NERBASE
  - Then run
    python3 -m stanza.utils.datasets.ner.prepare_ner_dataset lcode_masakhane
  - You can use the full language name, the 3 letter language code,
    or in the case of languages with a 2 letter language code,
    the 2 letter code for lcode.  The tool will throw an error
    if the language is not supported in Masakhane.

en_sample is the toy dataset included with stanza-train
  https://github.com/stanfordnlp/stanza-train
  this is not meant for any kind of actual NER use

"""

import glob
import os
import random
import re
import shutil
import sys
import tempfile

from stanza.models.common.constant import treebank_to_short_name, lcode2lang, lang_to_langcode, two_to_three_letters
import stanza.utils.default_paths as default_paths

from stanza.utils.datasets.ner.preprocess_wikiner import preprocess_wikiner
from stanza.utils.datasets.ner.split_wikiner import split_wikiner
import stanza.utils.datasets.ner.conll_to_iob as conll_to_iob
import stanza.utils.datasets.ner.convert_bn_daffodil as convert_bn_daffodil
import stanza.utils.datasets.ner.convert_bsf_to_beios as convert_bsf_to_beios
import stanza.utils.datasets.ner.convert_bsnlp as convert_bsnlp
import stanza.utils.datasets.ner.convert_fire_2013 as convert_fire_2013
import stanza.utils.datasets.ner.convert_ijc as convert_ijc
import stanza.utils.datasets.ner.convert_kk_kazNERD as convert_kk_kazNERD
import stanza.utils.datasets.ner.convert_lst20 as convert_lst20
import stanza.utils.datasets.ner.convert_mr_l3cube as convert_mr_l3cube
import stanza.utils.datasets.ner.convert_my_ucsy as convert_my_ucsy
import stanza.utils.datasets.ner.convert_rgai as convert_rgai
import stanza.utils.datasets.ner.convert_nytk as convert_nytk
import stanza.utils.datasets.ner.convert_starlang_ner as convert_starlang_ner
import stanza.utils.datasets.ner.convert_nkjp as convert_nkjp
import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file
import stanza.utils.datasets.ner.suc_to_iob as suc_to_iob
import stanza.utils.datasets.ner.suc_conll_to_iob as suc_conll_to_iob
from stanza.utils.datasets.ner.utils import convert_bio_to_json, get_tags, read_tsv, write_dataset

SHARDS = ('train', 'dev', 'test')

class UnknownDatasetError(ValueError):
    def __init__(self, dataset, text):
        super().__init__(text)
        self.dataset = dataset

def process_turku(paths, short_name):
    assert short_name == 'fi_turku'
    base_input_path = os.path.join(paths["NERBASE"], "finnish", "turku-ner-corpus", "data", "conll")
    base_output_path = paths["NER_DATA_DIR"]
    for shard in SHARDS:
        input_filename = os.path.join(base_input_path, '%s.tsv' % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def process_it_fbk(paths, short_name):
    assert short_name == "it_fbk"
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    csv_file = os.path.join(base_input_path, "all-wiki-split.tsv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError("Cannot find the FBK dataset in its expected location: {}".format(csv_file))
    base_output_path = paths["NER_DATA_DIR"]
    split_wikiner(base_output_path, csv_file, prefix=short_name, suffix="io", shuffle=False, train_fraction=0.8, dev_fraction=0.1)
    convert_bio_to_json(base_output_path, base_output_path, short_name, suffix="io")


def process_languk(paths, short_name):
    assert short_name == 'uk_languk'
    base_input_path = os.path.join(paths["NERBASE"], 'lang-uk', 'ner-uk', 'data')
    base_output_path = paths["NER_DATA_DIR"]
    train_test_split_fname = os.path.join(paths["NERBASE"], 'lang-uk', 'ner-uk', 'doc', 'dev-test-split.txt')
    convert_bsf_to_beios.convert_bsf_in_folder(base_input_path, base_output_path, train_test_split_file=train_test_split_fname)
    for shard in SHARDS:
        input_filename = os.path.join(base_output_path, convert_bsf_to_beios.CORPUS_NAME, "%s.bio" % shard)
        if not os.path.exists(input_filename):
            raise FileNotFoundError('Cannot find %s component of %s in %s' % (shard, short_name, input_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(input_filename, output_filename)


def process_ijc(paths, short_name):
    """
    Splits the ijc Hindi dataset in train, dev, test

    The original data had train & test splits, so we randomly divide
    the files in train to make a dev set.

    The expected location of the IJC data is hi_ijc.  This method
    should be possible to use for other languages, but we have very
    little support for the other languages of IJC at the moment.
    """
    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]

    test_files = [os.path.join(base_input_path, "test-data-hindi.txt")]
    test_csv_file = os.path.join(base_output_path, short_name + ".test.csv")
    print("Converting test input %s to space separated file in %s" % (test_files[0], test_csv_file))
    convert_ijc.convert_ijc(test_files, test_csv_file)

    train_input_path = os.path.join(base_input_path, "training-hindi", "*utf8")
    train_files = glob.glob(train_input_path)
    train_csv_file = os.path.join(base_output_path, short_name + ".train.csv")
    dev_csv_file = os.path.join(base_output_path, short_name + ".dev.csv")
    print("Converting training input from %s to space separated files in %s and %s" % (train_input_path, train_csv_file, dev_csv_file))
    convert_ijc.convert_split_ijc(train_files, train_csv_file, dev_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), SHARDS):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)


def process_fire_2013(paths, dataset):
    """
    Splits the FIRE 2013 dataset into train, dev, test

    The provided datasets are all mixed together at this point, so it
    is not possible to recreate the original test conditions used in
    the bakeoff
    """
    short_name = treebank_to_short_name(dataset)
    langcode, _ = short_name.split("_")
    short_name = "%s_fire2013" % langcode
    if not langcode in ("hi", "en", "ta", "bn", "mal"):
        raise UnkonwnDatasetError(dataset, "Language %s not one of the FIRE 2013 languages" % langcode)
    language = lcode2lang[langcode].lower()
    
    # for example, FIRE2013/hindi_train
    base_input_path = os.path.join(paths["NERBASE"], "FIRE2013", "%s_train" % language)
    base_output_path = paths["NER_DATA_DIR"]

    train_csv_file = os.path.join(base_output_path, "%s.train.csv" % short_name)
    dev_csv_file   = os.path.join(base_output_path, "%s.dev.csv" % short_name)
    test_csv_file  = os.path.join(base_output_path, "%s.test.csv" % short_name)

    convert_fire_2013.convert_fire_2013(base_input_path, train_csv_file, dev_csv_file, test_csv_file)

    for csv_file, shard in zip((train_csv_file, dev_csv_file, test_csv_file), SHARDS):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)

def process_wikiner(paths, dataset):
    short_name = treebank_to_short_name(dataset)

    base_input_path = os.path.join(paths["NERBASE"], dataset)
    base_output_path = paths["NER_DATA_DIR"]

    expected_filename = "aij*wikiner*"
    input_files = [x for x in glob.glob(os.path.join(base_input_path, expected_filename)) if not x.endswith("bz2")]
    if len(input_files) == 0:
        raw_input_path = os.path.join(base_input_path, "raw")
        input_files = [x for x in glob.glob(os.path.join(raw_input_path, expected_filename)) if not x.endswith("bz2")]
        if len(input_files) > 1:
            raise FileNotFoundError("Found too many raw wikiner files in %s: %s" % (raw_input_path, ", ".join(input_files)))
    elif len(input_files) > 1:
        raise FileNotFoundError("Found too many raw wikiner files in %s: %s" % (base_input_path, ", ".join(input_files)))

    if len(input_files) == 0:
        raise FileNotFoundError("Could not find any raw wikiner files in %s or %s" % (base_input_path, raw_input_path))

    csv_file = os.path.join(base_output_path, short_name + "_csv")
    print("Converting raw input %s to space separated file in %s" % (input_files[0], csv_file))
    try:
        preprocess_wikiner(input_files[0], csv_file)
    except UnicodeDecodeError:
        preprocess_wikiner(input_files[0], csv_file, encoding="iso8859-1")

    # this should create train.bio, dev.bio, and test.bio
    print("Splitting %s to %s" % (csv_file, base_output_path))
    split_wikiner(base_output_path, csv_file, prefix=short_name)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def get_rgai_input_path(paths):
    return os.path.join(paths["NERBASE"], "hu_rgai")

def process_rgai(paths, short_name):
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = get_rgai_input_path(paths)

    if short_name == 'hu_rgai':
        use_business = True
        use_criminal = True
    elif short_name == 'hu_rgai_business':
        use_business = True
        use_criminal = False
    elif short_name == 'hu_rgai_criminal':
        use_business = False
        use_criminal = True
    else:
        raise UnknownDatasetError(short_name, "Unknown subset of hu_rgai data: %s" % short_name)

    convert_rgai.convert_rgai(base_input_path, base_output_path, short_name, use_business, use_criminal)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def get_nytk_input_path(paths):
    return os.path.join(paths["NERBASE"], "NYTK-NerKor")

def process_nytk(paths, short_name):
    """
    Process the NYTK dataset
    """
    assert short_name == "hu_nytk"
    base_output_path = paths["NER_DATA_DIR"]
    base_input_path = get_nytk_input_path(paths)

    convert_nytk.convert_nytk(base_input_path, base_output_path, short_name)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def concat_files(output_file, *input_files):
    input_lines = []
    for input_file in input_files:
        with open(input_file) as fin:
            lines = fin.readlines()
        if not len(lines):
            raise ValueError("Empty input file: %s" % input_file)
        if not lines[-1]:
            lines[-1] = "\n"
        elif lines[-1].strip():
            lines.append("\n")
        input_lines.append(lines)
    with open(output_file, "w") as fout:
        for lines in input_lines:
            for line in lines:
                fout.write(line)


def process_hu_combined(paths, short_name):
    assert short_name == "hu_combined"

    base_output_path = paths["NER_DATA_DIR"]
    rgai_input_path = get_rgai_input_path(paths)
    nytk_input_path = get_nytk_input_path(paths)

    with tempfile.TemporaryDirectory() as tmp_output_path:
        convert_rgai.convert_rgai(rgai_input_path, tmp_output_path, "hu_rgai", True, True)
        convert_nytk.convert_nytk(nytk_input_path, tmp_output_path, "hu_nytk")

        for shard in SHARDS:
            rgai_input = os.path.join(tmp_output_path, "hu_rgai.%s.bio" % shard)
            nytk_input = os.path.join(tmp_output_path, "hu_nytk.%s.bio" % shard)
            output_file = os.path.join(base_output_path, "hu_combined.%s.bio" % shard)
            concat_files(output_file, rgai_input, nytk_input)

    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_bsnlp(paths, short_name):
    """
    Process files downloaded from http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html

    If you download the training and test data zip files and unzip
    them without rearranging in any way, the layout is somewhat weird.
    Training data goes into a specific subdirectory, but the test data
    goes into the top level directory.
    """
    base_input_path = os.path.join(paths["NERBASE"], "bsnlp2019")
    base_train_path = os.path.join(base_input_path, "training_pl_cs_ru_bg_rc1")
    base_test_path = base_input_path

    base_output_path = paths["NER_DATA_DIR"]

    output_train_filename = os.path.join(base_output_path, "%s.train.csv" % short_name)
    output_dev_filename   = os.path.join(base_output_path, "%s.dev.csv" % short_name)
    output_test_filename  = os.path.join(base_output_path, "%s.test.csv" % short_name)

    language = short_name.split("_")[0]

    convert_bsnlp.convert_bsnlp(language, base_test_path, output_test_filename)
    convert_bsnlp.convert_bsnlp(language, base_train_path, output_train_filename, output_dev_filename)

    for shard, csv_file in zip(SHARDS, (output_train_filename, output_dev_filename, output_test_filename)):
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, shard))
        prepare_ner_file.process_dataset(csv_file, output_filename)

NCHLT_LANGUAGE_MAP = {
    "af":  "NCHLT Afrikaans Named Entity Annotated Corpus",
    # none of the following have UD datasets as of 2.8.  Until they
    # exist, we assume the language codes NCHTL are sufficient
    "nr":  "NCHLT isiNdebele Named Entity Annotated Corpus",
    "nso": "NCHLT Sepedi Named Entity Annotated Corpus",
    "ss":  "NCHLT Siswati Named Entity Annotated Corpus",
    "st":  "NCHLT Sesotho Named Entity Annotated Corpus",
    "tn":  "NCHLT Setswana Named Entity Annotated Corpus",
    "ts":  "NCHLT Xitsonga Named Entity Annotated Corpus",
    "ve":  "NCHLT Tshivenda Named Entity Annotated Corpus",
    "xh":  "NCHLT isiXhosa Named Entity Annotated Corpus",
    "zu":  "NCHLT isiZulu Named Entity Annotated Corpus",
}

def process_nchlt(paths, short_name):
    language = short_name.split("_")[0]
    if not language in NCHLT_LANGUAGE_MAP:
        raise UnknownDatasetError(short_name, "Language %s not part of NCHLT" % language)
    short_name = "%s_nchlt" % language

    base_input_path = os.path.join(paths["NERBASE"], "NCHLT", NCHLT_LANGUAGE_MAP[language], "*Full.txt")
    input_files = glob.glob(base_input_path)
    if len(input_files) == 0:
        raise FileNotFoundError("Cannot find NCHLT dataset in '%s'  Did you remember to download the file?" % base_input_path)

    if len(input_files) > 1:
        raise ValueError("Unexpected number of files matched '%s'  There should only be one" % base_input_path)

    base_output_path = paths["NER_DATA_DIR"]
    split_wikiner(base_output_path, input_files[0], prefix=short_name, remap={"OUT": "O"})
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_my_ucsy(paths, short_name):
    assert short_name == "my_ucsy"
    language = "my"

    base_input_path = os.path.join(paths["NERBASE"], short_name)
    base_output_path = paths["NER_DATA_DIR"]
    convert_my_ucsy.convert_my_ucsy(base_input_path, base_output_path)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_fa_arman(paths, short_name):
    """
    Converts fa_arman dataset

    The conversion is quite simple, actually.
    Just need to split the train file and then convert bio -> json
    """
    assert short_name == "fa_arman"
    language = "fa"
    base_input_path = os.path.join(paths["NERBASE"], "PersianNER")
    train_input_file = os.path.join(base_input_path, "train_fold1.txt")
    test_input_file = os.path.join(base_input_path, "test_fold1.txt")
    if not os.path.exists(train_input_file) or not os.path.exists(test_input_file):
        full_corpus_file = os.path.join(base_input_path, "ArmanPersoNERCorpus.zip")
        if os.path.exists(full_corpus_file):
            raise FileNotFoundError("Please unzip the file {}".format(full_corpus_file))
        raise FileNotFoundError("Cannot find the arman corpus in the expected directory: {}".format(base_input_path))

    base_output_path = paths["NER_DATA_DIR"]
    test_output_file = os.path.join(base_output_path, "%s.test.bio" % short_name)

    split_wikiner(base_output_path, train_input_file, prefix=short_name, train_fraction=0.8, test_section=False)
    shutil.copy2(test_input_file, test_output_file)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_sv_suc3licensed(paths, short_name):
    """
    The .zip provided for SUC3 includes train/dev/test splits already

    This extracts those splits without needing to unzip the original file
    """
    assert short_name == "sv_suc3licensed"
    language = "sv"
    train_input_file = os.path.join(paths["NERBASE"], short_name, "SUC3.0.zip")
    if not os.path.exists(train_input_file):
        raise FileNotFoundError("Cannot find the officially licensed SUC3 dataset in %s" % train_input_file)

    base_output_path = paths["NER_DATA_DIR"]
    suc_conll_to_iob.process_suc3(train_input_file, short_name, base_output_path)
    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_sv_suc3shuffle(paths, short_name):
    """
    Uses an externally provided script to read the SUC3 XML file, then splits it
    """
    assert short_name == "sv_suc3shuffle"
    language = "sv"
    train_input_file = os.path.join(paths["NERBASE"], short_name, "suc3.xml.bz2")
    if not os.path.exists(train_input_file):
        train_input_file = train_input_file[:-4]
    if not os.path.exists(train_input_file):
        raise FileNotFoundError("Unable to find the SUC3 dataset in {}.bz2".format(train_input_file))

    base_output_path = paths["NER_DATA_DIR"]
    train_output_file = os.path.join(base_output_path, "sv_suc3shuffle.bio")
    suc_to_iob.main([train_input_file, train_output_file])
    split_wikiner(base_output_path, train_output_file, prefix=short_name)
    convert_bio_to_json(base_output_path, base_output_path, short_name)    
    
def process_da_ddt(paths, short_name):
    """
    Processes Danish DDT dataset

    This dataset is in a conll file with the "name" attribute in the
    misc column for the NER tag.  This function uses a script to
    convert such CoNLL files to .bio
    """
    assert short_name == "da_ddt"
    language = "da"
    IN_FILES = ("ddt.train.conllu", "ddt.dev.conllu", "ddt.test.conllu")

    base_output_path = paths["NER_DATA_DIR"]
    OUT_FILES = [os.path.join(base_output_path, "%s.%s.bio" % (short_name, shard)) for shard in SHARDS]

    zip_file = os.path.join(paths["NERBASE"], "da_ddt", "ddt.zip")
    if os.path.exists(zip_file):
        for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
            conll_to_iob.process_conll(in_filename, out_filename, zip_file)
    else:
        for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
            in_filename = os.path.join(paths["NERBASE"], "da_ddt", in_filename)
            if not os.path.exists(in_filename):
                raise FileNotFoundError("Could not find zip in expected location %s and could not file %s file in %s" % (zip_file, shard, in_filename))

            conll_to_iob.process_conll(in_filename, out_filename)
    convert_bio_to_json(base_output_path, base_output_path, short_name)


def process_norne(paths, short_name):
    """
    Processes Norwegian NorNE

    Can handle either Bokmål or Nynorsk

    Converts GPE_LOC and GPE_ORG to GPE
    """
    language, name = short_name.split("_", 1)
    assert language in ('nb', 'nn')
    assert name == 'norne'

    if language == 'nb':
        IN_FILES = ("nob/no_bokmaal-ud-train.conllu", "nob/no_bokmaal-ud-dev.conllu", "nob/no_bokmaal-ud-test.conllu")
    else:
        IN_FILES = ("nno/no_nynorsk-ud-train.conllu", "nno/no_nynorsk-ud-dev.conllu", "nno/no_nynorsk-ud-test.conllu")

    base_output_path = paths["NER_DATA_DIR"]
    OUT_FILES = [os.path.join(base_output_path, "%s.%s.bio" % (short_name, shard)) for shard in SHARDS]

    CONVERSION = { "GPE_LOC": "GPE", "GPE_ORG": "GPE" }

    for in_filename, out_filename, shard in zip(IN_FILES, OUT_FILES, SHARDS):
        in_filename = os.path.join(paths["NERBASE"], "norne", "ud", in_filename)
        if not os.path.exists(in_filename):
            raise FileNotFoundError("Could not find %s file in %s" % (shard, in_filename))

        conll_to_iob.process_conll(in_filename, out_filename, conversion=CONVERSION)

    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_ja_gsd(paths, short_name):
    """
    Convert ja_gsd from MegagonLabs

    for example, can download from https://github.com/megagonlabs/UD_Japanese-GSD/releases/tag/r2.9-NE
    """
    language, name = short_name.split("_", 1)
    assert language == 'ja'
    assert name == 'gsd'

    base_output_path = paths["NER_DATA_DIR"]
    output_files = [os.path.join(base_output_path, "%s.%s.bio" % (short_name, shard)) for shard in SHARDS]

    search_path = os.path.join(paths["NERBASE"], "ja_gsd", "UD_Japanese-GSD-r2.*-NE")
    versions = glob.glob(search_path)
    max_version = None
    base_input_path = None
    version_re = re.compile("GSD-r2.([0-9]+)-NE$")

    for ver in versions:
        match = version_re.search(ver)
        if not match:
            continue
        ver_num = int(match.groups(1)[0])
        if max_version is None or ver_num > max_version:
            max_version = ver_num
            base_input_path = ver

    if base_input_path is None:
        raise FileNotFoundError("Could not find any copies of the NE conversion of ja_gsd here: {}".format(search_path))
    print("Most recent version found: {}".format(base_input_path))

    input_files = ["ja_gsd-ud-train.ne.conllu", "ja_gsd-ud-dev.ne.conllu", "ja_gsd-ud-test.ne.conllu"]

    def conversion(x):
        if x[0] == 'L':
            return 'E' + x[1:]
        if x[0] == 'U':
            return 'S' + x[1:]
        # B, I unchanged
        return x

    for in_filename, out_filename, shard in zip(input_files, output_files, SHARDS):
        in_path = os.path.join(base_input_path, in_filename)
        if not os.path.exists(in_path):
            in_spacy = os.path.join(base_input_path, "spacy", in_filename)
            if not os.path.exists(in_spacy):
                raise FileNotFoundError("Could not find %s file in %s or %s" % (shard, in_path, in_spacy))
            in_path = in_spacy

        conll_to_iob.process_conll(in_path, out_filename, conversion=conversion, allow_empty=True, attr_prefix="NE")

    convert_bio_to_json(base_output_path, base_output_path, short_name)

def process_starlang(paths, short_name):
    """
    Process a Turkish dataset from Starlang
    """
    assert short_name == 'tr_starlang'

    PIECES = ["TurkishAnnotatedTreeBank-15",
              "TurkishAnnotatedTreeBank2-15",
              "TurkishAnnotatedTreeBank2-20"]

    chunk_paths = [os.path.join(paths["CONSTITUENCY_BASE"], "turkish", piece) for piece in PIECES]
    datasets = convert_starlang_ner.read_starlang(chunk_paths)

    write_dataset(datasets, paths["NER_DATA_DIR"], short_name)

def remap_germeval_tag(tag):
    """
    Simplify tags for GermEval2014 using a simple rubric

    all tags become their parent tag
    OTH becomes MISC
    """
    if tag == "O":
        return tag
    if tag[1:5] == "-LOC":
        return tag[:5]
    if tag[1:5] == "-PER":
        return tag[:5]
    if tag[1:5] == "-ORG":
        return tag[:5]
    if tag[1:5] == "-OTH":
        return tag[0] + "-MISC"
    raise ValueError("Unexpected tag: %s" % tag)

def process_de_germeval2014(paths, short_name):
    """
    Process the TSV of the GermEval2014 dataset
    """
    in_directory = os.path.join(paths["NERBASE"], "germeval2014")
    base_output_path = paths["NER_DATA_DIR"]
    datasets = []
    for shard in SHARDS:
        in_file = os.path.join(in_directory, "NER-de-%s.tsv" % shard)
        sentences = read_tsv(in_file, 1, 2, remap_fn=remap_germeval_tag)
        datasets.append(sentences)
    tags = get_tags(datasets)
    print("Found the following tags: {}".format(sorted(tags)))
    write_dataset(datasets, base_output_path, short_name)

def process_hiner(paths, short_name):
    in_directory = os.path.join(paths["NERBASE"], "hindi", "HiNER", "data", "original")
    convert_bio_to_json(in_directory, paths["NER_DATA_DIR"], short_name, suffix="conll", shard_names=("train", "validation", "test"))

def process_hinercollapsed(paths, short_name):
    in_directory = os.path.join(paths["NERBASE"], "hindi", "HiNER", "data", "collapsed")
    convert_bio_to_json(in_directory, paths["NER_DATA_DIR"], short_name, suffix="conll", shard_names=("train", "validation", "test"))

def process_lst20(paths, short_name, include_space_char=True):
    convert_lst20.convert_lst20(paths, short_name, include_space_char)

def process_mr_l3cube(paths, short_name):
    base_output_path = paths["NER_DATA_DIR"]
    in_directory = os.path.join(paths["NERBASE"], "marathi", "MarathiNLP", "L3Cube-MahaNER", "IOB")
    input_files = ["train_iob.txt", "valid_iob.txt", "test_iob.txt"]
    input_files = [os.path.join(in_directory, x) for x in input_files]
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError("Could not find the expected piece of the l3cube dataset %s" % input_file)

    datasets = [convert_mr_l3cube.convert(input_file) for input_file in input_files]
    write_dataset(datasets, base_output_path, short_name)

def process_bn_daffodil(paths, short_name):
    in_directory = os.path.join(paths["NERBASE"], "bangla", "Bengali-NER")
    out_directory = paths["NER_DATA_DIR"]
    convert_bn_daffodil.convert_dataset(in_directory, out_directory)

def process_pl_nkjp(paths, short_name):
    out_directory = paths["NER_DATA_DIR"]
    candidates = [os.path.join(paths["NERBASE"], "Polish-NKJP"),
                  os.path.join(paths["NERBASE"], "polish", "Polish-NKJP"),
                  os.path.join(paths["NERBASE"], "polish", "NKJP-PodkorpusMilionowy-1.2.tar.gz"),]
    for in_path in candidates:
        if os.path.exists(in_path):
            break
    else:
        raise FileNotFoundError("Could not find %s  Looked in %s" % (short_name, " ".join(candidates)))
    convert_nkjp.convert_nkjp(in_path, out_directory)

def process_kk_kazNERD(paths, short_name):
    in_directory = os.path.join(paths["NERBASE"], "kazakh", "KazNERD", "KazNERD")
    out_directory = paths["NER_DATA_DIR"]
    convert_kk_kazNERD.convert_dataset(in_directory, out_directory, short_name)

def process_masakhane(paths, dataset_name):
    """
    Converts Masakhane NER datasets to Stanza's .json format

    If we let N be the length of the first sentence, the NER files
    (in version 2, at least) are all of the form

    word tag
    ...
    word tag
      (blank line for sentence break)
    word tag
    ...

    Once the dataset is git cloned in $NERBASE, the directory structure is

    $NERBASE/masakhane-ner/MasakhaNER2.0/data/$lcode/{train,dev,test}.txt

    The only tricky thing here is that for some languages, we treat
    the 2 letter lcode as canonical thanks to UD, but Masakhane NER
    uses 3 letter lcodes for all languages.
    """
    language, dataset = dataset_name.split("_")
    lcode = lang_to_langcode(language)
    if lcode in two_to_three_letters:
        masakhane_lcode = two_to_three_letters[lcode]
    else:
        masakhane_lcode = lcode

    mn_directory = os.path.join(paths["NERBASE"], "masakhane-ner")
    if not os.path.exists(mn_directory):
        raise FileNotFoundError("Cannot find Masakhane NER repo.  Please check the setting of NERBASE or clone the repo to %s" % mn_directory)
    data_directory = os.path.join(mn_directory, "MasakhaNER2.0", "data")
    if not os.path.exists(data_directory):
        raise FileNotFoundError("Apparently found the repo at %s but the expected directory structure is not there - was looking for %s" % (mn_directory, data_directory))

    in_directory = os.path.join(data_directory, masakhane_lcode)
    if not os.path.exists(in_directory):
        raise UnknownDatasetError(dataset_name, "Found the Masakhane repo, but there was no %s in the repo at path %s" % (dataset_name, in_directory))
    convert_bio_to_json(in_directory, paths["NER_DATA_DIR"], "%s_masakhane" % lcode, "txt")

def process_toy_dataset(paths, short_name):
    convert_bio_to_json(os.path.join(paths["NERBASE"], "English-SAMPLE"), paths["NER_DATA_DIR"], short_name)

DATASET_MAPPING = {
    "bn_daffodil":       process_bn_daffodil,
    "da_ddt":            process_da_ddt,
    "de_germeval2014":   process_de_germeval2014,
    "fa_arman":          process_fa_arman,
    "fi_turku":          process_turku,
    "hi_hiner":          process_hiner,
    "hi_hinercollapsed": process_hinercollapsed,
    "hi_ijc":            process_ijc,
    "hu_nytk":           process_nytk,
    "hu_combined":       process_hu_combined,
    "it_fbk":            process_it_fbk,
    "ja_gsd":            process_ja_gsd,
    "kk_kazNERD":        process_kk_kazNERD,
    "mr_l3cube":         process_mr_l3cube,
    "my_ucsy":           process_my_ucsy,
    "pl_nkjp":           process_pl_nkjp,
    "sv_suc3licensed":   process_sv_suc3licensed,
    "sv_suc3shuffle":    process_sv_suc3shuffle,
    "tr_starlang":       process_starlang,
    "th_lst20":          process_lst20,
}

def main(dataset_name):
    paths = default_paths.get_default_paths()
    print("Processing %s" % dataset_name)

    random.seed(1234)

    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name)
    elif dataset_name in ('uk_languk', 'Ukranian_languk', 'Ukranian-languk'):
        process_languk(paths, dataset_name)
    elif dataset_name.endswith("FIRE2013") or dataset_name.endswith("fire2013"):
        process_fire_2013(paths, dataset_name)
    elif dataset_name.endswith('WikiNER'):
        process_wikiner(paths, dataset_name)
    elif dataset_name.startswith('hu_rgai'):
        process_rgai(paths, dataset_name)
    elif dataset_name.endswith("_bsnlp19"):
        process_bsnlp(paths, dataset_name)
    elif dataset_name.endswith("_nchlt"):
        process_nchlt(paths, dataset_name)
    elif dataset_name in ("nb_norne", "nn_norne"):
        process_norne(paths, dataset_name)
    elif dataset_name == 'en_sample':
        process_toy_dataset(paths, dataset_name)
    elif dataset_name.lower().endswith("_masakhane"):
        process_masakhane(paths, dataset_name)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_ner_dataset")
    print("Done processing %s" % dataset_name)

if __name__ == '__main__':
    main(sys.argv[1])
