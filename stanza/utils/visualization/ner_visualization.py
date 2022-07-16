from spacy import displacy
from spacy.tokens import Doc
from spacy.tokens import Span
from stanza.models.common.constant import is_right_to_left
import stanza
import spacy
import copy


def visualize_ner_doc(doc, language, select=None, colors=None):
    """
    Takes a stanza doc object and language pipeline and visualizes the named entities within it.

    Stanza currently supports a limited amount of languages for NER, which you can view here:
    https://stanfordnlp.github.io/stanza/ner_models.html

    To view only a specific type(s) of named entities, set the optional 'select' argument to
    a list of the named entity types. Ex: select=["PER", "ORG", "GPE"] to only see entities tagged as Person(s),
    Organizations, and Geo-political entities. A full list of the available types can be found here:
    https://stanfordnlp.github.io/stanza/ner_models.html (ctrl + F "The following table").

    The colors argument is formatted as a dictionary of NER tags with their corresponding colors, which can be
    represented as a string (ex: "blue"), a color hex value (ex: #aa9cfc), or as a linear gradient of color
    values (ex: "linear-gradient(90deg, #aa9cfc, #fc9ce7)").

    Do not change the 'rtl_clr_adjusted' argument; it is used for ensuring that the visualize_strings function
    works properly on rtl languages.
    """
    model, documents, visualization_colors = spacy.blank('en'), [], copy.deepcopy(colors)  # blank model, spacy is only used for visualization purposes
    sentences, rtl, RTL_OVERRIDE = doc.sentences, is_right_to_left(language), "‮"
    if rtl:  # need to flip order of all the sentences in rendered display
        sentences = reversed(doc.sentences)
        # adjust colors to be in LTR flipped format due to the RLO unicode char flipping words
        if colors:
            for color in visualization_colors:
                if RTL_OVERRIDE not in color:
                    clr_val = visualization_colors[color]
                    visualization_colors.pop(color)
                    visualization_colors[RTL_OVERRIDE + color[::-1]] = clr_val
    for sentence in sentences:
        words, display_ents, already_found = [], [], False
        # initialize doc object with words first
        for i, word in enumerate(sentence.words):
            if rtl and word.text.isascii() and not already_found:
                to_append = [word.text[::-1]]
                next_word_index = i + 1
                # account for flipping non Arabic words back to original form and order. two flips -> original order
                while next_word_index <= len(sentence.words) - 1 and sentence.words[next_word_index].text.isascii():
                    to_append.append(sentence.words[next_word_index].text[::-1])
                    next_word_index += 1
                to_append = reversed(to_append)
                for token in to_append:
                    words.append(token)
                already_found = True
            elif rtl and word.text.isascii() and already_found:  # skip over already collected words
                continue
            else:  # arabic chars
                words.append(word.text)
                already_found = False

        document = Doc(model.vocab, words=words)

        # tag all NER tokens found
        for ent in sentence.ents:
            if select and ent.type not in select:
                continue
            found_indexes = []
            for token in ent.tokens:
                found_indexes.append(token.id[0] - 1)
            if not rtl:
                to_add = Span(document, found_indexes[0], found_indexes[-1] + 1, ent.type)
            else:  # RTL languages need the override char to flip order
                to_add = Span(document, found_indexes[0], found_indexes[-1] + 1, RTL_OVERRIDE + ent.type[::-1])
            display_ents.append(to_add)
        document.set_ents(display_ents)
        documents.append(document)

    # Visualize doc objects
    visualization_options = {"ents": select}
    if colors:
        visualization_options["colors"] = visualization_colors
    for document in documents:
        displacy.render(document, style='ent', options=visualization_options)


def visualize_ner_str(text, pipe, select=None, colors=None):
    """
    Takes in a text string and visualizes the named entities within the text.

    Required args also include a pipeline code, the two-letter code for a language defined by Universal Dependencies (ex: "en" for English).

    Lastly, the user must provide an NLP pipeline - we recommend Stanza (ex: pipe = stanza.Pipeline('en')).

    Optionally, the 'select' argument allows for specific NER tags to be highlighted; the 'color' argument allows
    for specific NER tags to have certain color(s).
    """
    doc = pipe(text)
    visualize_ner_doc(doc, pipe.lang, select, colors)


def visualize_strings(texts, language_code, select=None, colors=None):
    """
    Takes in a list of strings and a language code (Stanza defines these, ex: 'en' for English) to visualize all
    of the strings' named entities.

    The strings are processed by the Stanza pipeline and the named entities are displayed. Each text is separated by a delimiting line.

    Optionally, the 'select' argument may be configured to only visualize given named entities (ex: select=['ORG', 'PERSON']).

    The optional colors argument is formatted as a dictionary of NER tags with their corresponding colors, which can be
    represented as a string (ex: "blue"), a color hex value (ex: #aa9cfc), or as a linear gradient of color
    values (ex: "linear-gradient(90deg, #aa9cfc, #fc9ce7)").
    """
    lang_pipe = stanza.Pipeline(language_code, processors="tokenize,ner")

    for text in texts:
        visualize_ner_str(text, lang_pipe, select=select, colors=colors)


def visualize_docs(docs, language_code, select=None, colors=None):
    """
    Takes in a list of doc and a language code (Stanza defines these, ex: 'en' for English) to visualize all
    of the strings' named entities.

    Each text is separated by a delimiting line.

    Optionally, the 'select' argument may be configured to only visualize given named entities (ex: select=['ORG', 'PERSON']).

    The optional colors argument is formatted as a dictionary of NER tags with their corresponding colors, which can be
    represented as a string (ex: "blue"), a color hex value (ex: #aa9cfc), or as a linear gradient of color
    values (ex: "linear-gradient(90deg, #aa9cfc, #fc9ce7)").
    """
    for doc in docs:
        visualize_ner_doc(doc, language_code, select=select, colors=colors)


def main():
    en_strings = ['''Samuel Jackson, a Christian man from Utah, went to the JFK Airport for a flight to New York.
                               He was thinking of attending the US Open, his favorite tennis tournament besides Wimbledon.
                               That would be a dream trip, certainly not possible since it is $5000 attendance and 5000 miles away.
                               On the way there, he watched the Super Bowl for 2 hours and read War and Piece by Tolstoy for 1 hour.
                               In New York, he crossed the Brooklyn Bridge and listened to the 5th symphony of Beethoven as well as
                               "All I want for Christmas is You" by Mariah Carey.''',
                  "Barack Obama was born in Hawaii. He was elected President of the United States in 2008"]
    zh_strings = ['''来自犹他州的基督徒塞缪尔杰克逊前往肯尼迪机场搭乘航班飞往纽约。
                             他正在考虑参加美国公开赛，这是除了温布尔登之外他最喜欢的网球赛事。
                             那将是一次梦想之旅，当然不可能，因为它的出勤费为 5000 美元，距离 5000 英里。
                             在去的路上，他看了 2 个小时的超级碗比赛，看了 1 个小时的托尔斯泰的《战争与碎片》。
                               在纽约，他穿过布鲁克林大桥，聆听了贝多芬的第五交响曲以及 玛丽亚凯莉的“圣诞节我想要的就是你”。''',
                  "我觉得罗家费德勒住在加州, 在美国里面。"]
    ar_strings = [
        ".أعيش في سان فرانسيسكو ، كاليفورنيا. اسمي أليكس وأنا ألتحق بجامعة ستانفورد. أنا أدرس علوم الكمبيوتر وأستاذي هو كريس مانينغ"
        , "اسمي أليكس ، أنا من الولايات المتحدة.",
        '''صامويل جاكسون ، رجل مسيحي من ولاية يوتا ، ذهب إلى مطار جون كنيدي في رحلة إلى نيويورك. كان يفكر في حضور بطولة الولايات المتحدة المفتوحة للتنس ، بطولة التنس المفضلة لديه إلى جانب بطولة ويمبلدون. ستكون هذه رحلة الأحلام ، وبالتأكيد ليست ممكنة لأنها تبلغ 5000 دولار للحضور و 5000 ميل. في الطريق إلى هناك ، شاهد Super Bowl لمدة ساعتين وقرأ War and Piece by Tolstoy لمدة ساعة واحدة. في نيويورك ، عبر جسر بروكلين واستمع إلى السيمفونية الخامسة لبيتهوفن وكذلك "كل ما أريده في عيد الميلاد هو أنت" لماريا كاري.''']

    visualize_strings(en_strings, "en")
    visualize_strings(zh_strings, "zh", colors={"PERSON": "yellow", "DATE": "red", "GPE": "blue"})
    visualize_strings(zh_strings, "zh", select=['PERSON', 'DATE'])
    visualize_strings(ar_strings, "ar",
                      colors={"PER": "pink", "LOC": "linear-gradient(90deg, #aa9cfc, #fc9ce7)", "ORG": "yellow"})


if __name__ == "__main__":
    main()
