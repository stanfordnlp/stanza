import pytest

import io
import os
import xml.etree.ElementTree as ET

from stanza.utils.datasets.ner.convert_nkjp import MORPH_FILE, NER_FILE, extract_entities_from_subfolder, extract_entities_from_sentence, extract_unassigned_subfolder_entities

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EXPECTED_ENTITIES = {
    '1-p': {
        '1.39-s': [{'ent_id': 'named_1.39-s_n1', 'index': 0, 'orth': 'Sił Zbrojnych', 'ner_type': 'orgName', 'ner_subtype': None, 'targets': ['1.37-seg', '1.38-seg']}],
        '1.56-s': [],
        '1.79-s': []
    },
    '2-p': {
        '2.30-s': [],
        '2.45-s': []
    },
    '3-p': {
        '3.70-s': []
    }
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    dataset_path = tmp_path_factory.mktemp("nkjp_dataset")
    sample_path = dataset_path / "sample"
    os.mkdir(sample_path)
    ann_path = sample_path / NER_FILE
    with open(ann_path, "w", encoding="utf-8") as fout:
        fout.write(SAMPLE_ANN)
    morph_path = sample_path / MORPH_FILE
    with open(morph_path, "w", encoding="utf-8") as fout:
        fout.write(SAMPLE_MORPHO)
    return dataset_path

EXPECTED_TOKENS = [
    {'seg_id': '1.1-seg', 'i': 0, 'orth': '2', 'text': '2', 'tag': '_', 'ner': 'O', 'ner_subtype': None},
    {'seg_id': '1.37-seg', 'i': 36, 'orth': 'Sił', 'text': 'Sił', 'tag': '_', 'ner': 'B-orgName', 'ner_subtype': None},
    {'seg_id': '1.38-seg', 'i': 37, 'orth': 'Zbrojnych', 'text': 'Zbrojnych', 'tag': '_', 'ner': 'I-orgName', 'ner_subtype': None},
]

def test_extract_entities_from_subfolder(dataset):
    entities = extract_entities_from_subfolder("sample", dataset)
    assert len(entities) == 1
    assert len(entities['1-p']) == 1
    assert len(entities['1-p']['1.39-s']) == 39
    assert entities['1-p']['1.39-s']['1.1-seg'] == EXPECTED_TOKENS[0]
    assert entities['1-p']['1.39-s']['1.37-seg'] == EXPECTED_TOKENS[1]
    assert entities['1-p']['1.39-s']['1.38-seg'] == EXPECTED_TOKENS[2]


def test_extract_unassigned(dataset):
    entities = extract_unassigned_subfolder_entities("sample", dataset)
    assert entities == EXPECTED_ENTITIES

SENTENCE_SAMPLE = """
          <s xmlns="http://www.tei-c.org/ns/1.0" xmlns:xi="http://www.w3.org/2001/XInclude" xml:id="named_1.39-s" corresp="ann_morphosyntax.xml#morph_1.39-s">
            <seg xml:id="named_1.39-s_n1">
              <fs type="named">
                <f name="type">
                  <symbol value="orgName"/>
                </f>
                <f name="orth">
                  <string>Si&#322; Zbrojnych</string>
                </f>
                <f name="base">
                  <string>Si&#322;y Zbrojne</string>
                </f>
                <f name="certainty">
                  <symbol value="high"/>
                </f>
              </fs>
              <ptr target="ann_morphosyntax.xml#morph_1.37-seg"/>
              <ptr target="ann_morphosyntax.xml#morph_1.38-seg"/>
            </seg>
          </s>
""".strip()


EMPTY_SENTENCE = """<s xml:id="named_1.56-s" corresp="ann_morphosyntax.xml#morph_1.56-s"/>"""

def test_extract_entities_from_sentence():
    rt = ET.fromstring(SENTENCE_SAMPLE)
    entities = extract_entities_from_sentence(rt)
    assert entities == EXPECTED_ENTITIES['1-p']['1.39-s']

    rt = ET.fromstring(EMPTY_SENTENCE)
    entities = extract_entities_from_sentence(rt)
    assert entities == []



# picked completely at random, one sample file for testing:
# 610-1-000248/ann_named.xml
# only the first sentence is used in the morpho file
SAMPLE_ANN = """
<?xml version='1.0' encoding='UTF-8'?>
<teiCorpus xmlns:xi="http://www.w3.org/2001/XInclude" xmlns="http://www.tei-c.org/ns/1.0">
  <xi:include href="NKJP_1M_header.xml"/>
  <TEI>
    <xi:include href="header.xml"/>
    <text xml:lang="pl">
      <body>
        <p xml:id="named_1-p" corresp="ann_morphosyntax.xml#morph_1-p">
          <s xml:id="named_1.39-s" corresp="ann_morphosyntax.xml#morph_1.39-s">
            <seg xml:id="named_1.39-s_n1">
              <fs type="named">
                <f name="type">
                  <symbol value="orgName"/>
                </f>
                <f name="orth">
                  <string>Sił Zbrojnych</string>
                </f>
                <f name="base">
                  <string>Siły Zbrojne</string>
                </f>
                <f name="certainty">
                  <symbol value="high"/>
                </f>
              </fs>
              <ptr target="ann_morphosyntax.xml#morph_1.37-seg"/>
              <ptr target="ann_morphosyntax.xml#morph_1.38-seg"/>
            </seg>
          </s>
          <s xml:id="named_1.56-s" corresp="ann_morphosyntax.xml#morph_1.56-s"/>
          <s xml:id="named_1.79-s" corresp="ann_morphosyntax.xml#morph_1.79-s"/>
        </p>
        <p xml:id="named_2-p" corresp="ann_morphosyntax.xml#morph_2-p">
          <s xml:id="named_2.30-s" corresp="ann_morphosyntax.xml#morph_2.30-s"/>
          <s xml:id="named_2.45-s" corresp="ann_morphosyntax.xml#morph_2.45-s"/>
        </p>
        <p xml:id="named_3-p" corresp="ann_morphosyntax.xml#morph_3-p">
          <s xml:id="named_3.70-s" corresp="ann_morphosyntax.xml#morph_3.70-s"/>
        </p>
      </body>
    </text>
  </TEI>
</teiCorpus>
""".lstrip()



SAMPLE_MORPHO = """
<?xml version="1.0" encoding="UTF-8"?>
<!-- w indeksach elementów wciąganych mogą zdarzyć się nieciągłości (z alternatyw segmentacyjnych)  --><teiCorpus xmlns="http://www.tei-c.org/ns/1.0" xmlns:nkjp="http://www.nkjp.pl/ns/1.0" xmlns:xi="http://www.w3.org/2001/XInclude">
 <xi:include href="NKJP_1M_header.xml"/>
 <TEI>
  <xi:include href="header.xml"/>
  <text>
   <body>
    <!-- morph_1-p is akapit 7626 with instances (akapit_transzy-s) 15244, 15269 in batches (transza-s) 1525, 1528 resp. -->
    <p corresp="ann_segmentation.xml#segm_1-p" xml:id="morph_1-p">
     <s corresp="ann_segmentation.xml#segm_1.39-s" xml:id="morph_1.39-s">
      <seg corresp="ann_segmentation.xml#segm_1.1-seg" xml:id="morph_1.1-seg">
       <fs type="morph">
        <f name="orth">
         <string>2</string>
        </f>
        <!-- 2 [0,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.1.1-lex">
          <f name="base">
           <string/>
          </f>
          <f name="ctag">
           <symbol value="ign"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.1.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.1.2-lex">
          <f name="base">
           <string>2</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <symbol nkjp:manual="true" value="sg:nom:n:pos" xml:id="morph_1.1.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.1.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>2:adj:sg:nom:n:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.2-seg" xml:id="morph_1.2-seg">
       <fs type="morph">
        <f name="orth">
         <string>.</string>
        </f>
        <!-- . [1,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.2.1-lex">
          <f name="base">
           <string>.</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.2.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.2.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>.:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.3-seg" xml:id="morph_1.3-seg">
       <fs type="morph">
        <f name="orth">
         <string>Wezwanie</string>
        </f>
        <!-- Wezwanie [3,8] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.3.1-lex">
          <f name="base">
           <string>wezwanie</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:nom:n" xml:id="morph_1.3.1.1-msd"/>
            <symbol value="sg:acc:n" xml:id="morph_1.3.1.2-msd"/>
            <symbol value="sg:voc:n" xml:id="morph_1.3.1.3-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.3.2-lex">
          <f name="base">
           <string>wezwać</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:nom:n:perf:aff" xml:id="morph_1.3.2.1-msd"/>
            <symbol value="sg:acc:n:perf:aff" xml:id="morph_1.3.2.2-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.3.1.2-msd" name="choice"/>
          <f name="interpretation">
           <string>wezwanie:subst:sg:acc:n</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.4-seg" xml:id="morph_1.4-seg">
       <fs type="morph">
        <f name="orth">
         <string>,</string>
        </f>
        <!-- , [11,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.4.1-lex">
          <f name="base">
           <string>,</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.4.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.4.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>,:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.5-seg" xml:id="morph_1.5-seg">
       <fs type="morph">
        <f name="orth">
         <string>o</string>
        </f>
        <!-- o [13,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.5.1-lex">
          <f name="base">
           <string>o</string>
          </f>
          <f name="ctag">
           <symbol value="interj"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.5.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.5.2-lex">
          <f name="base">
           <string>o</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc" xml:id="morph_1.5.2.1-msd"/>
            <symbol value="loc" xml:id="morph_1.5.2.2-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.5.3-lex">
          <f name="base">
           <string>ojciec</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.5.3.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.5.2.2-msd" name="choice"/>
          <f name="interpretation">
           <string>o:prep:loc</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.6-seg" xml:id="morph_1.6-seg">
       <fs type="morph">
        <f name="orth">
         <string>którym</string>
        </f>
        <!-- którym [15,6] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.6.1-lex">
          <f name="base">
           <string>który</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:inst:m1:pos" xml:id="morph_1.6.1.1-msd"/>
            <symbol value="sg:inst:m2:pos" xml:id="morph_1.6.1.2-msd"/>
            <symbol value="sg:inst:m3:pos" xml:id="morph_1.6.1.3-msd"/>
            <symbol value="sg:inst:n:pos" xml:id="morph_1.6.1.4-msd"/>
            <symbol value="sg:loc:m1:pos" xml:id="morph_1.6.1.5-msd"/>
            <symbol value="sg:loc:m2:pos" xml:id="morph_1.6.1.6-msd"/>
            <symbol value="sg:loc:m3:pos" xml:id="morph_1.6.1.7-msd"/>
            <symbol value="sg:loc:n:pos" xml:id="morph_1.6.1.8-msd"/>
            <symbol value="pl:dat:m1:pos" xml:id="morph_1.6.1.9-msd"/>
            <symbol value="pl:dat:m2:pos" xml:id="morph_1.6.1.10-msd"/>
            <symbol value="pl:dat:m3:pos" xml:id="morph_1.6.1.11-msd"/>
            <symbol value="pl:dat:f:pos" xml:id="morph_1.6.1.12-msd"/>
            <symbol value="pl:dat:n:pos" xml:id="morph_1.6.1.13-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.6.1.8-msd" name="choice"/>
          <f name="interpretation">
           <string>który:adj:sg:loc:n:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.7-seg" xml:id="morph_1.7-seg">
       <fs type="morph">
        <f name="orth">
         <string>mowa</string>
        </f>
        <!-- mowa [22,4] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.7.1-lex">
          <f name="base">
           <string>mowa</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="sg:nom:f" xml:id="morph_1.7.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.7.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>mowa:subst:sg:nom:f</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.8-seg" xml:id="morph_1.8-seg">
       <fs type="morph">
        <f name="orth">
         <string>w</string>
        </f>
        <!-- w [27,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.8.1-lex">
          <f name="base">
           <string>w</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc:nwok" xml:id="morph_1.8.1.1-msd"/>
            <symbol value="loc:nwok" xml:id="morph_1.8.1.2-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.8.2-lex">
          <f name="base">
           <string>wiek</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.8.2.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.8.3-lex">
          <f name="base">
           <string>wielki</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.8.3.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.8.4-lex">
          <f name="base">
           <string>wiersz</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.8.4.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.8.5-lex">
          <f name="base">
           <string>wieś</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.8.5.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.8.6-lex">
          <f name="base">
           <string>wyspa</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.8.6.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.8.1.2-msd" name="choice"/>
          <f name="interpretation">
           <string>w:prep:loc:nwok</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.9-seg" xml:id="morph_1.9-seg">
       <fs type="morph">
        <f name="orth">
         <string>ust</string>
        </f>
        <!-- ust [29,3] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.9.1-lex">
          <f name="base">
           <string>usta</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="pl:gen:n" xml:id="morph_1.9.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.9.2-lex">
          <f name="base">
           <string>ustęp</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol nkjp:manual="true" value="pun" xml:id="morph_1.9.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.9.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>ustęp:brev:pun</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.10-seg" xml:id="morph_1.10-seg">
       <fs type="morph">
        <f name="orth">
         <string>.</string>
        </f>
        <!-- . [32,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.10.1-lex">
          <f name="base">
           <string>.</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.10.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.10.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>.:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.11-seg" xml:id="morph_1.11-seg">
       <fs type="morph">
        <f name="orth">
         <string>1</string>
        </f>
        <!-- 1 [34,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.11.1-lex">
          <f name="base">
           <string/>
          </f>
          <f name="ctag">
           <symbol value="ign"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.11.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.11.2-lex">
          <f name="base">
           <string>1</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <symbol nkjp:manual="true" value="sg:loc:m3:pos" xml:id="morph_1.11.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.11.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>1:adj:sg:loc:m3:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.12-seg" xml:id="morph_1.12-seg">
       <fs type="morph">
        <f name="orth">
         <string>,</string>
        </f>
        <!-- , [35,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.12.1-lex">
          <f name="base">
           <string>,</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.12.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.12.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>,:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.13-seg" xml:id="morph_1.13-seg">
       <fs type="morph">
        <f name="orth">
         <string>doręcza</string>
        </f>
        <!-- doręcza [37,7] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.13.1-lex">
          <f name="base">
           <string>doręczać</string>
          </f>
          <f name="ctag">
           <symbol value="fin"/>
          </f>
          <f name="msd">
           <symbol value="sg:ter:imperf" xml:id="morph_1.13.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.13.2-lex">
          <f name="base">
           <string>doręcze</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:n" xml:id="morph_1.13.2.1-msd"/>
            <symbol value="pl:nom:n" xml:id="morph_1.13.2.2-msd"/>
            <symbol value="pl:acc:n" xml:id="morph_1.13.2.3-msd"/>
            <symbol value="pl:voc:n" xml:id="morph_1.13.2.4-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.13.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>doręczać:fin:sg:ter:imperf</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.14-seg" xml:id="morph_1.14-seg">
       <fs type="morph">
        <f name="orth">
         <string>się</string>
        </f>
        <!-- się [45,3] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.14.1-lex">
          <f name="base">
           <string>się</string>
          </f>
          <f name="ctag">
           <symbol value="qub"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.14.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.14.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>się:qub</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.15-seg" xml:id="morph_1.15-seg">
       <fs type="morph">
        <f name="orth">
         <string>na</string>
        </f>
        <!-- na [49,2] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.15.1-lex">
          <f name="base">
           <string>na</string>
          </f>
          <f name="ctag">
           <symbol value="interj"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.15.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.15.2-lex">
          <f name="base">
           <string>na</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc" xml:id="morph_1.15.2.1-msd"/>
            <symbol value="loc" xml:id="morph_1.15.2.2-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.15.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>na:prep:acc</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.16-seg" xml:id="morph_1.16-seg">
       <fs type="morph">
        <f name="orth">
         <string>czternaście</string>
        </f>
        <!-- czternaście [52,11] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.16.1-lex">
          <f name="base">
           <string>czternaście</string>
          </f>
          <f name="ctag">
           <symbol value="num"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="pl:nom:m2:rec" xml:id="morph_1.16.1.1-msd"/>
            <symbol value="pl:nom:m3:rec" xml:id="morph_1.16.1.2-msd"/>
            <symbol value="pl:nom:f:rec" xml:id="morph_1.16.1.3-msd"/>
            <symbol value="pl:nom:n:rec" xml:id="morph_1.16.1.4-msd"/>
            <symbol value="pl:acc:m2:rec" xml:id="morph_1.16.1.5-msd"/>
            <symbol value="pl:acc:m3:rec" xml:id="morph_1.16.1.6-msd"/>
            <symbol value="pl:acc:f:rec" xml:id="morph_1.16.1.7-msd"/>
            <symbol value="pl:acc:n:rec" xml:id="morph_1.16.1.8-msd"/>
            <symbol value="pl:voc:m2:rec" xml:id="morph_1.16.1.9-msd"/>
            <symbol value="pl:voc:m3:rec" xml:id="morph_1.16.1.10-msd"/>
            <symbol value="pl:voc:f:rec" xml:id="morph_1.16.1.11-msd"/>
            <symbol value="pl:voc:n:rec" xml:id="morph_1.16.1.12-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.16.1.6-msd" name="choice"/>
          <f name="interpretation">
           <string>czternaście:num:pl:acc:m3:rec</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.17-seg" xml:id="morph_1.17-seg">
       <fs type="morph">
        <f name="orth">
         <string>dni</string>
        </f>
        <!-- dni [64,3] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.17.1-lex">
          <f name="base">
           <string>dni</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="pl:nom:n" xml:id="morph_1.17.1.1-msd"/>
            <symbol value="pl:gen:n" xml:id="morph_1.17.1.2-msd"/>
            <symbol value="pl:acc:n" xml:id="morph_1.17.1.3-msd"/>
            <symbol value="pl:voc:n" xml:id="morph_1.17.1.4-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.17.2-lex">
          <f name="base">
           <string>dzień</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="pl:nom:m3" xml:id="morph_1.17.2.1-msd"/>
            <symbol value="pl:gen:m3" xml:id="morph_1.17.2.2-msd"/>
            <symbol value="pl:acc:m3" xml:id="morph_1.17.2.3-msd"/>
            <symbol value="pl:voc:m3" xml:id="morph_1.17.2.4-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.17.2.2-msd" name="choice"/>
          <f name="interpretation">
           <string>dzień:subst:pl:gen:m3</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.18-seg" xml:id="morph_1.18-seg">
       <fs type="morph">
        <f name="orth">
         <string>przed</string>
        </f>
        <!-- przed [68,5] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.18.1-lex">
          <f name="base">
           <string>przed</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc:nwok" xml:id="morph_1.18.1.1-msd"/>
            <symbol value="inst:nwok" xml:id="morph_1.18.1.2-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.18.1.2-msd" name="choice"/>
          <f name="interpretation">
           <string>przed:prep:inst:nwok</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.19-seg" xml:id="morph_1.19-seg">
       <fs type="morph">
        <f name="orth">
         <string>terminem</string>
        </f>
        <!-- terminem [74,8] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.19.1-lex">
          <f name="base">
           <string>termin</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="sg:inst:m3" xml:id="morph_1.19.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.19.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>termin:subst:sg:inst:m3</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.20-seg" xml:id="morph_1.20-seg">
       <fs type="morph">
        <f name="orth">
         <string>wykonania</string>
        </f>
        <!-- wykonania [83,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.20.1-lex">
          <f name="base">
           <string>wykonanie</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:n" xml:id="morph_1.20.1.1-msd"/>
            <symbol value="pl:nom:n" xml:id="morph_1.20.1.2-msd"/>
            <symbol value="pl:acc:n" xml:id="morph_1.20.1.3-msd"/>
            <symbol value="pl:voc:n" xml:id="morph_1.20.1.4-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.20.2-lex">
          <f name="base">
           <string>wykonać</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <symbol value="sg:gen:n:perf:aff" xml:id="morph_1.20.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.20.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>wykonać:ger:sg:gen:n:perf:aff</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.21-seg" xml:id="morph_1.21-seg">
       <fs type="morph">
        <f name="orth">
         <string>świadczenia</string>
        </f>
        <!-- świadczenia [93,11] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.21.1-lex">
          <f name="base">
           <string>świadczenie</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:n" xml:id="morph_1.21.1.1-msd"/>
            <symbol value="pl:nom:n" xml:id="morph_1.21.1.2-msd"/>
            <symbol value="pl:acc:n" xml:id="morph_1.21.1.3-msd"/>
            <symbol value="pl:voc:n" xml:id="morph_1.21.1.4-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.21.2-lex">
          <f name="base">
           <string>świadczyć</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <symbol value="sg:gen:n:imperf:aff" xml:id="morph_1.21.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.21.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>świadczenie:subst:sg:gen:n</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.22-seg" xml:id="morph_1.22-seg">
       <fs type="morph">
        <f name="orth">
         <string>,</string>
        </f>
        <!-- , [104,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.22.1-lex">
          <f name="base">
           <string>,</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.22.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.22.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>,:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.23-seg" xml:id="morph_1.23-seg">
       <fs type="morph">
        <f name="orth">
         <string>z</string>
        </f>
        <!-- z [106,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.23.1-lex">
          <f name="base">
           <string>z</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="gen:nwok" xml:id="morph_1.23.1.1-msd"/>
            <symbol value="acc:nwok" xml:id="morph_1.23.1.2-msd"/>
            <symbol value="inst:nwok" xml:id="morph_1.23.1.3-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.23.2-lex">
          <f name="base">
           <string>z</string>
          </f>
          <f name="ctag">
           <symbol value="qub"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.23.2.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.23.3-lex">
          <f name="base">
           <string>zeszyt</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.23.3.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.23.1.3-msd" name="choice"/>
          <f name="interpretation">
           <string>z:prep:inst:nwok</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.24-seg" xml:id="morph_1.24-seg">
       <fs type="morph">
        <f name="orth">
         <string>wyjątkiem</string>
        </f>
        <!-- wyjątkiem [108,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.24.1-lex">
          <f name="base">
           <string>wyjątek</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="sg:inst:m3" xml:id="morph_1.24.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.24.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>wyjątek:subst:sg:inst:m3</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.25-seg" xml:id="morph_1.25-seg">
       <fs type="morph">
        <f name="orth">
         <string>przypadków</string>
        </f>
        <!-- przypadków [118,10] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.25.1-lex">
          <f name="base">
           <string>przypadek</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="pl:gen:m3" xml:id="morph_1.25.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.25.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>przypadek:subst:pl:gen:m3</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.26-seg" xml:id="morph_1.26-seg">
       <fs type="morph">
        <f name="orth">
         <string>,</string>
        </f>
        <!-- , [128,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.26.1-lex">
          <f name="base">
           <string>,</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.26.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.26.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>,:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.27-seg" xml:id="morph_1.27-seg">
       <fs type="morph">
        <f name="orth">
         <string>w</string>
        </f>
        <!-- w [130,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.27.1-lex">
          <f name="base">
           <string>w</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc:nwok" xml:id="morph_1.27.1.1-msd"/>
            <symbol value="loc:nwok" xml:id="morph_1.27.1.2-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.27.2-lex">
          <f name="base">
           <string>wiek</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.27.2.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.27.3-lex">
          <f name="base">
           <string>wielki</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.27.3.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.27.4-lex">
          <f name="base">
           <string>wiersz</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.27.4.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.27.5-lex">
          <f name="base">
           <string>wieś</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.27.5.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.27.6-lex">
          <f name="base">
           <string>wyspa</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.27.6.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.27.1.2-msd" name="choice"/>
          <f name="interpretation">
           <string>w:prep:loc:nwok</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.28-seg" xml:id="morph_1.28-seg">
       <fs type="morph">
        <f name="orth">
         <string>których</string>
        </f>
        <!-- których [132,7] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.28.1-lex">
          <f name="base">
           <string>który</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="pl:gen:m1:pos" xml:id="morph_1.28.1.1-msd"/>
            <symbol value="pl:gen:m2:pos" xml:id="morph_1.28.1.2-msd"/>
            <symbol value="pl:gen:m3:pos" xml:id="morph_1.28.1.3-msd"/>
            <symbol value="pl:gen:f:pos" xml:id="morph_1.28.1.4-msd"/>
            <symbol value="pl:gen:n:pos" xml:id="morph_1.28.1.5-msd"/>
            <symbol value="pl:loc:m1:pos" xml:id="morph_1.28.1.6-msd"/>
            <symbol value="pl:loc:m2:pos" xml:id="morph_1.28.1.7-msd"/>
            <symbol value="pl:loc:m3:pos" xml:id="morph_1.28.1.8-msd"/>
            <symbol value="pl:loc:f:pos" xml:id="morph_1.28.1.9-msd"/>
            <symbol value="pl:loc:n:pos" xml:id="morph_1.28.1.10-msd"/>
            <symbol value="pl:acc:m1:pos" xml:id="morph_1.28.1.11-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.28.1.8-msd" name="choice"/>
          <f name="interpretation">
           <string>który:adj:pl:loc:m3:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.29-seg" xml:id="morph_1.29-seg">
       <fs type="morph">
        <f name="orth">
         <string>wykonanie</string>
        </f>
        <!-- wykonanie [140,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.29.1-lex">
          <f name="base">
           <string>wykonanie</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:nom:n" xml:id="morph_1.29.1.1-msd"/>
            <symbol value="sg:acc:n" xml:id="morph_1.29.1.2-msd"/>
            <symbol value="sg:voc:n" xml:id="morph_1.29.1.3-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.29.2-lex">
          <f name="base">
           <string>wykonać</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:nom:n:perf:aff" xml:id="morph_1.29.2.1-msd"/>
            <symbol value="sg:acc:n:perf:aff" xml:id="morph_1.29.2.2-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.29.2.1-msd" name="choice"/>
          <f name="interpretation">
           <string>wykonać:ger:sg:nom:n:perf:aff</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.30-seg" xml:id="morph_1.30-seg">
       <fs type="morph">
        <f name="orth">
         <string>świadczenia</string>
        </f>
        <!-- świadczenia [150,11] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.30.1-lex">
          <f name="base">
           <string>świadczenie</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:n" xml:id="morph_1.30.1.1-msd"/>
            <symbol value="pl:nom:n" xml:id="morph_1.30.1.2-msd"/>
            <symbol value="pl:acc:n" xml:id="morph_1.30.1.3-msd"/>
            <symbol value="pl:voc:n" xml:id="morph_1.30.1.4-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.30.2-lex">
          <f name="base">
           <string>świadczyć</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <symbol value="sg:gen:n:imperf:aff" xml:id="morph_1.30.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.30.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>świadczenie:subst:sg:gen:n</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.31-seg" xml:id="morph_1.31-seg">
       <fs type="morph">
        <f name="orth">
         <string>następuje</string>
        </f>
        <!-- następuje [162,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.31.1-lex">
          <f name="base">
           <string>następować</string>
          </f>
          <f name="ctag">
           <symbol value="fin"/>
          </f>
          <f name="msd">
           <symbol value="sg:ter:imperf" xml:id="morph_1.31.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.31.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>następować:fin:sg:ter:imperf</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.32-seg" xml:id="morph_1.32-seg">
       <fs type="morph">
        <f name="orth">
         <string>w</string>
        </f>
        <!-- w [172,1] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.32.1-lex">
          <f name="base">
           <string>w</string>
          </f>
          <f name="ctag">
           <symbol value="prep"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="acc:nwok" xml:id="morph_1.32.1.1-msd"/>
            <symbol value="loc:nwok" xml:id="morph_1.32.1.2-msd"/>
           </vAlt>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.32.2-lex">
          <f name="base">
           <string>wiek</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.32.2.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.32.3-lex">
          <f name="base">
           <string>wielki</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.32.3.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.32.4-lex">
          <f name="base">
           <string>wiersz</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.32.4.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.32.5-lex">
          <f name="base">
           <string>wieś</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.32.5.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.32.6-lex">
          <f name="base">
           <string>wyspa</string>
          </f>
          <f name="ctag">
           <symbol value="brev"/>
          </f>
          <f name="msd">
           <symbol value="pun" xml:id="morph_1.32.6.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.32.1.2-msd" name="choice"/>
          <f name="interpretation">
           <string>w:prep:loc:nwok</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.33-seg" xml:id="morph_1.33-seg">
       <fs type="morph">
        <f name="orth">
         <string>celu</string>
        </f>
        <!-- celu [174,4] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.33.1-lex">
          <f name="base">
           <string>Cela</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="sg:voc:f" xml:id="morph_1.33.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.33.2-lex">
          <f name="base">
           <string>cel</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:m3" xml:id="morph_1.33.2.1-msd"/>
            <symbol value="sg:loc:m3" xml:id="morph_1.33.2.2-msd"/>
            <symbol value="sg:voc:m3" xml:id="morph_1.33.2.3-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.33.2.2-msd" name="choice"/>
          <f name="interpretation">
           <string>cel:subst:sg:loc:m3</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.34-seg" xml:id="morph_1.34-seg">
       <fs type="morph">
        <f name="orth">
         <string>sprawdzenia</string>
        </f>
        <!-- sprawdzenia [179,11] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.34.1-lex">
          <f name="base">
           <string>sprawdzić</string>
          </f>
          <f name="ctag">
           <symbol value="ger"/>
          </f>
          <f name="msd">
           <symbol value="sg:gen:n:perf:aff" xml:id="morph_1.34.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.34.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>sprawdzić:ger:sg:gen:n:perf:aff</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.35-seg" xml:id="morph_1.35-seg">
       <fs type="morph">
        <f name="orth">
         <string>gotowości</string>
        </f>
        <!-- gotowości [191,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.35.1-lex">
          <f name="base">
           <string>gotowość</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:f" xml:id="morph_1.35.1.1-msd"/>
            <symbol value="sg:dat:f" xml:id="morph_1.35.1.2-msd"/>
            <symbol value="sg:loc:f" xml:id="morph_1.35.1.3-msd"/>
            <symbol value="sg:voc:f" xml:id="morph_1.35.1.4-msd"/>
            <symbol value="pl:nom:f" xml:id="morph_1.35.1.5-msd"/>
            <symbol value="pl:gen:f" xml:id="morph_1.35.1.6-msd"/>
            <symbol value="pl:acc:f" xml:id="morph_1.35.1.7-msd"/>
            <symbol value="pl:voc:f" xml:id="morph_1.35.1.8-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.35.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>gotowość:subst:sg:gen:f</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.36-seg" xml:id="morph_1.36-seg">
       <fs type="morph">
        <f name="orth">
         <string>mobilizacyjnej</string>
        </f>
        <!-- mobilizacyjnej [201,14] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.36.1-lex">
          <f name="base">
           <string>mobilizacyjny</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="sg:gen:f:pos" xml:id="morph_1.36.1.1-msd"/>
            <symbol value="sg:dat:f:pos" xml:id="morph_1.36.1.2-msd"/>
            <symbol value="sg:loc:f:pos" xml:id="morph_1.36.1.3-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.36.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>mobilizacyjny:adj:sg:gen:f:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.37-seg" xml:id="morph_1.37-seg">
       <fs type="morph">
        <f name="orth">
         <string>Sił</string>
        </f>
        <!-- Sił [216,3] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.37.1-lex">
          <f name="base">
           <string>siła</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="pl:gen:f" xml:id="morph_1.37.1.1-msd"/>
          </f>
         </fs>
         <fs type="lex" xml:id="morph_1.37.2-lex">
          <f name="base">
           <string>siły</string>
          </f>
          <f name="ctag">
           <symbol value="subst"/>
          </f>
          <f name="msd">
           <symbol value="pl:gen:n" xml:id="morph_1.37.2.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.37.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>siła:subst:pl:gen:f</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.38-seg" xml:id="morph_1.38-seg">
       <fs type="morph">
        <f name="orth">
         <string>Zbrojnych</string>
        </f>
        <!-- Zbrojnych [220,9] -->
        <f name="interps">
         <fs type="lex" xml:id="morph_1.38.1-lex">
          <f name="base">
           <string>zbrojny</string>
          </f>
          <f name="ctag">
           <symbol value="adj"/>
          </f>
          <f name="msd">
           <vAlt>
            <symbol value="pl:gen:m1:pos" xml:id="morph_1.38.1.1-msd"/>
            <symbol value="pl:gen:m2:pos" xml:id="morph_1.38.1.2-msd"/>
            <symbol value="pl:gen:m3:pos" xml:id="morph_1.38.1.3-msd"/>
            <symbol value="pl:gen:f:pos" xml:id="morph_1.38.1.4-msd"/>
            <symbol value="pl:gen:n:pos" xml:id="morph_1.38.1.5-msd"/>
            <symbol value="pl:loc:m1:pos" xml:id="morph_1.38.1.6-msd"/>
            <symbol value="pl:loc:m2:pos" xml:id="morph_1.38.1.7-msd"/>
            <symbol value="pl:loc:m3:pos" xml:id="morph_1.38.1.8-msd"/>
            <symbol value="pl:loc:f:pos" xml:id="morph_1.38.1.9-msd"/>
            <symbol value="pl:loc:n:pos" xml:id="morph_1.38.1.10-msd"/>
            <symbol value="pl:acc:m1:pos" xml:id="morph_1.38.1.11-msd"/>
           </vAlt>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.38.1.4-msd" name="choice"/>
          <f name="interpretation">
           <string>zbrojny:adj:pl:gen:f:pos</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
      <seg corresp="ann_segmentation.xml#segm_1.39-seg" xml:id="morph_1.39-seg">
       <fs type="morph">
        <f name="orth">
         <string>.</string>
        </f>
        <!-- . [229,1] -->
        <f name="nps">
         <binary value="true"/>
        </f>
        <f name="interps">
         <fs type="lex" xml:id="morph_1.39.1-lex">
          <f name="base">
           <string>.</string>
          </f>
          <f name="ctag">
           <symbol value="interp"/>
          </f>
          <f name="msd">
           <symbol value="" xml:id="morph_1.39.1.1-msd"/>
          </f>
         </fs>
        </f>
        <f name="disamb">
         <fs feats="#an8003" type="tool_report">
          <f fVal="#morph_1.39.1.1-msd" name="choice"/>
          <f name="interpretation">
           <string>.:interp</string>
          </f>
         </fs>
        </f>
       </fs>
      </seg>
     </s>
    </p>
   </body>
  </text>
 </TEI>
</teiCorpus>
""".lstrip()
