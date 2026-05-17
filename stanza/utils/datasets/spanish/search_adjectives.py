import argparse
from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL

# caliza?  not sure if this is a special case or not

# NOT an invariable demonym:
# brucia
# attested here https://www.scribd.com/document/245042707/Brucios

no_feature_cases = [
    "anti",
    "ex",
    "ultra",
    "lumpen",
    "pop",
    "redox",
    "súper",
    "aparte",
    "antidopaje",   # as in 'controles antidopaje', which is much more accepted than 'controles antidopajes'
]

special_cases = [
    "alienígena",
    "indígena",
    "hipócrita",
    "idiota",
    "pirata",
    "extra",
    "quejica",  # whiny, apparently no quejico
    "psicópata",
    "rubioceniza", # a compound of two other ADJ
    "cosmopolita",
    "mosca", # annoyed
    "fantasma",
    "ultra",
    "nómada",
    "entusiasta",
    "boricua",
    "antidroga",
    "iconoclasta",
    "antisistema",
    "patriota",
    "okupa",    # not sure... is both M/F as NOUN, presumably also as ADJ?
    "ratonera", # rat trap... ratonero is a buzzard, so if used as ADJ, probably has to keep -a?
    "déspota",
    "biplaza",
    "lumpen",
    "toquilla",   # a type of straw, used as "straw hat"
    "masoca",     # masochist
    "hembra",     # female
    "punta",      # invariable for "peak velocity" etc?
    "multimedia",
    "basura",     # used in a phrase like "comida basura" as an ADJ
    "profamilia",
    "tierra-tierra",   # is this even a single word?  ground to ground (as in, missile)
    "supermosca",  # for example, in the context "campeón supermosca", where it is not inflected
    "antipersona", # minas antipersonas.  looked for "antipersono" and could not find anything like it
    # such as in the phrase "sangre mangosta", "mongoose blood", such
    # as you would inject into someone to make them a mongoose-powered superhero
    "mangosta",
    # in context: "el hijo honoriscausa".  not inflected in the text.
    # not in any dictionary.  described by ChatGPT as a Latin insertion.
    "honoriscausa",
    # specifically the Colombian dish "bandeja paisa"
    # this is getting in the weeds.  apparently there is also a plural form
    # "bandejas paisas" but i can't find a "paiso" attested to
    "paisa",

    # many demonyms are common gender / invariable for gender
    "persa",
    "belga",
    "francobelga",
    "croata",
    "hitita",
    "chiíta",
    "marroquí",
    "iraní",
    "iraquí",
    "israelí",
    "hindú",
    "zulú",
    "lisboeta",
    "blaugrana", # barcelona fan
    "azulgrana",
    # from San Salvador.  An example sentence in the online dictionary I found was
    #  "Juego en un equipo donostiarra"
    #  https://www.spanishdict.com/translate/donostiarra
    "donostiarra",
    "vietnamita",
    "celta",
    "maya",
    "moscovita",
    "tlaxcalteca",
    "chipriota",   # from Cyprus
    "carioca",     # from Rio de Janiero
    "inca",
    "azteca",
    "olmeca",
    "jesuíta",
    "purépecha",   # a specific Mexican minority
    "quechua",
    "quichua",
    "angora",
    "nahua",
    "etarra",   # member of ETA
    "myanma",
    "maronita", # Maronite Church in Lebanon
    "aymara",   # an Indigenous group of the Andes
    "semita",   # semite, has only -a forms
    "antisemita",
    "pastún",
    "hindú",
    # also not sure of.  does "omeyo" exist?
    "omeya",
    # perj. for gay
    "marica",


    "crema",    # the color cream, used as ADJ
    "rosa",
    "naranja",
    "violeta",
    "lavanda",
    "ultravioleta",
]

known_non_participles = set([
    "cordado",   # heart shaped
    "lanceolado",
    "peciolado",
    "pedunculado",
    "pinnado",  # pinnate

    "aqueménido",  # Achaemenid
    "númido",   # Numidian
    "timúrido",  # from timúr

    "ácido",
    "álgido",
    "árido",
    "ávido",
    "bienvenido",
    "cálido",
    "cándido",
    "centígrado",
    "consabido",  # no word consaber?
    "delgado",
    "delicado",
    "demasiado",
    "desafortunado",
    "desarrapado",
    "desmazalado",
    "espléndido",
    "estúpido",
    "frígido",
    "gélido",
    "híbrido",  # https://github.com/UniversalDependencies/UD_Spanish-GSD/issues/87
    "inadecuado",
    "intrépido",
    "inválido",
    "líquido",
    "lúcido",
    "malintencionado",
    "malvado",
    "morado",
    "nítido",
    "pálido",
    "pútrido",
    "rápido",
    "retrógrado",
    "rígido",
    "salmonado", # as in, salmon colored
    "semisólido",
    "sólido",
    "sórdido",
    "tímido",
    "válido",
    "vívido",

    "colimorada", # a type of hummingbird
    "nómada",

    "apelusada",

    "genocida", # -cida endings don't inflect
    "herbicida",
    "homicida",
    "suicida",
])

known_participles = set([
    "contradicho",
    "dicho",
    "predicho",

    "antepuesto",
    "compuesto",
    "contrapuesto",
    "depuesto",
    "dispuesto",
    "expuesto",
    "impuesto",
    "indispuesto",   # TODO does this exist?  different sources disagree
    "interpuesto",
    "opuesto",
    "pospuesto",
    "propuesto",
    "puesto",
    "repuesto",
    "sobrepuesto",
    "supuesto",

    "entrevisto", # entrever
    "previsto",
    "provisto",   # irregular form of proveer
    "visto",

    "abierto",
    "cubierto",
    "descubierto",
    "encubierto",
    "recubierto",

    # -ído ending, may need to search for other outliers
    "atraído",
    "caído",
    "contraído",
    "creído",
    "desvaído",
    "distraído",
    "engreído",
    "extraído",
    "leído",
    "oído",
    "poseído",
    "retraído",
    "reído",
    "roído",
    "sonreído",
    "sustraído",
    "traído",

    "hecho",
    "deshecho",
    "rehecho",
    "satisfecho",

    "devuelto",
    "envuelto",
    "revuelto",
    "vuelto",
    "absuelto",
    "disuelto",
    "resuelto",

    "adscrito",
    "circunscrito",
    "descrito",
    "escrito",
    "inscrito",
    "prescrito",
    "proscrito",
    "suscrito",
    "transcrito",

    "bendito",
    "maldito",

    "exento",

    "impreso",
    "reimpreso",
    "sobreimpreso",

    "frito",
    "refrito",

    "muerto",

    "roto",

    "asustado",
    "cantado",
    "educado",
    "enrazado",
    "formado",
    "guardado",
    "inclinado",
    "intercomunicado",
    "secado",
    "sembrado",
    "separado",
    "sobreasado",

    "agradecido",
    "aparecido",
    "cocido",
    "conducido",
    "conocido",
    "convencido",
    "coproducido",
    "desaparecido",
    "desconocido",
    "desvanecido",
    "ejercido",
    "empobrecido",
    "enaltecido",
    "enardecido",
    "enloquecido",
    "enrojecido",
    "entorpecido",
    "entristecido"
    "envejecido",
    "esclarecido",
    "escocido",
    "establecido",
    "fallecido",
    "inmunizado",
    "introducido",
    "lucido",
    "merecido",
    "nacido",
    "ofrecido",
    "oscurecido",
    "parecido",
    "perdido",
    "producido",
    "reconocido",
    "reducido",
    "reproducido",
    "retorcido",
    "reverdecido",
    "unido",
    "vencido",

    # starting here are only partially reviewed Part from GSD
    "abanderado",
    "abandonado",
    "abreviado",
    "abultado",
    "aburrido",
    "acantonado",
    "accidentado",
    "acelerado",
    "aclamado",
    "acodado",
    "acogido",
    "acomodado",
    "acompañado",
    "acondicionado",
    "acoplado",
    "acorazado",
    "acordado",
    "acorralado",
    "acostumbrado",
    "acristalado",
    "actualizado",
    "acumulado",
    "acusado",
    "adaptado",
    "adecuado",
    "adelantado",
    "adinerado",
    "admirado",
    "adornado",
    "adosado",
    "adquirido",
    "afamado",
    "afectado",
    "aficoinado",
    "afilado",
    "afiliado",
    "afinado",
    "afincado",
    "agarrado",
    "agigantado",
    "agitado",
    "agotado",
    "agrandado",
    "agravado",
    "aguerrido",
    "agujereado",
    "ahumado",
    "aislado",
    "ajado",
    "ajustado",
    "alargado",
    "alcanzado",
    "alejado",
    "alforzado",
    "aliado",
    "aficionado",
    "alienado",
    "almacenado",
    "alojado",
    "alterado",
    "alternado",
    "ambientado",
    "amenazado",
    "ampliado",
    "animado",
    "ansiado",
    "anticipado",
    "anticuado",
    "anualizado",
    "apisonado",
    "aplicado",
    "apocado",
    "apreciado",
    "aprehendido",
    "apresado",
    "apretado",
    "apropiado",
    "aproximado",
    "aquilatado",
    "arbolado",
    "armado",
    "arrasado",
    "arrastrado",
    "arreglado",
    "arrestado",
    "arriesgado",
    "arrugado",
    "arruinado",
    "articulado",
    "asado",
    "aseado",
    "asegurado",
    "asentado",
    "asociado",
    "aspirado",
    "asqueado",
    "atacado",
    "atado",
    "atascado",
    "atendido",
    "aterciopelado",
    "aterrazado",
    "atinado",
    "atrapado",
    "atrasado",
    "atrevido",
    "atribuido",
    "aturdido",
    "aumentado",
    "autoexigido",
    "autointeresado",
    "automatizado",
    "autotitulado",
    "avanzado",
    "aventajado",
    "azorado",
    "azulado",
    "añadido",
    "balanceado",
    "banalizado",
    "barrado",
    "basado",
    "bendecido",
    "beneficiado",
    "bicromatado",
    "blindado",
    "bloqueado",
    "bordado",
    "bronceado",
    "calado",
    "calibrado",
    "cansado",
    "capacitado",
    "capado",
    "capturado",
    "carenciado",
    "cargado",
    "casado",
    "cautivado",
    "cedido",
    "celebrado",
    "centrado",
    "centralizado",
    "cerrado",
    "citado",
    "civilizado",
    "codiciado",
    "cohesionado",
    "colmado",
    "colocado",
    "colorido",
    "combinado",
    "comedido",
    "comentado",
    "comercializado",
    "cometido",
    "compaginado",
    "compartido",
    "completado",
    "complicado",
    "comprado",
    "comprendido",
    "comprimido",
    "comprometido",
    "concentrado",
    "concertado",
    "concluido",
    "concurrido",
    "condenado",
    "condensado",
    "condicionado",
    "conectado",
    "confiado",
    "confluido",
    "confontado",
    "consagrado",
    "conseguido",
    "conservado",
    "considerado",
    "consolidado",
    "construido",
    "consultado",
    "contado",
    "contaminado",
    "contenido",
    "continuado",
    "contrastado",
    "contratado",
    "controlado",
    "controvertido",
    "convocado",
    "convulsionado",
    "coordinado",
    "cordado",
    "coreado",
    "coreografiado",
    "cortado",
    "creado",
    "crecido",
    "crispado",
    "crucificado",
    "cruzado",
    "cuadrado",
    "cuadriculado",
    "cualificado",
    "cuestionado",
    "cuidado",
    "cumplido",
    "curado",
    "dañado",
    "decepcionado",
    "decidido",
    "declinado",
    "decorado",
    "dedicado",
    "defendido",
    "definido",
    "deformado",
    "degradado",
    "delimitado",
    "demandado",
    "demostrado",
    "denodado",
    "denominado",
    "dentado",
    "denticulado",
    "deportado",
    "deprimido",
    "derivado",
    "derrotado",
    "derruido",
    "desahogado",
    "desahuciado",
    "desaliñado",
    "desapercibido",
    "desarrollado",
    "desatado",
    "descafeinado",
    "descartado",
    "descentralizado",
    "desconchado",
    "descontrolado",
    "deseado",
    "desenvainado",
    "desequilibrado",
    "desesperado",
    "desfasado",
    "desfigurado",
    "deshabitado",
    "designado",
    "deslavazado",
    "desmayado",
    "desmentido",
    "desocupado",
    "desordenado",
    "despavorido",
    "despedido",
    "despoblado",
    "desproporcionado",
    "desregulado",
    "destacado",
    "destinado",
    "destruido",
    "detallado",
    "detenido",
    "determinado",
    "devenido",
    "diferenciado",
    "difundido",
    "dilapidado",
    "dilatado",
    "dirigido",
    "discapacitado",
    "disciplinado",
    "diseminado",
    "disfrazado",
    "disminuido",
    "disputado",
    "distendido",
    "distinguido",
    "diversificado",
    "divertido",
    "dividido",
    "divorciado",
    "doblado",
    "documentado",
    "dorado",
    "dormido",
    "dotado",
    "drapeado",
    "duplicado",
    "elaborado",
    "elegido",
    "elevado",
    "embalado",
    "embaldosado",
    "embalsado",
    "embarazado",
    "embebido",
    "emocionado",
    "empapado",
    "emparentado",
    "empatado",
    "empeñado",
    "empinado",
    "emplazado",
    "enamorado",
    "encantado",
    "encaramado",
    "encargado",
    "encarnizado",
    "encendido",
    "enchufado",
    "encomendado",
    "enfadado",
    "enfatizado",
    "enfocado",
    "enlatado",
    "enmarcado",
    "enmoquetado",
    "enquistado",
    "enrevesado",
    "enriquecido",
    "entallado",
    "entretenido",
    "equilibrado",
    "equipado",
    "erguido",
    "erosionado",
    "escalonado",
    "escamado",
    "escarmentado",
    "escarpado",
    "escondido",
    "escotado",
    "esmaltado",
    "espaciado",
    "espantado",
    "especializado",
    "esperado",
    "estacionado",
    "estafado",
    "estancado",
    "estandarizado",
    "estereotipado",
    "esterilizado",
    "estilizado",
    "estratificado",
    "estropeado",
    "estudiado",
    "excitado",
    "excluido",
    "exiliado",
    "expandido",
    "experimentado",
    "explicado",
    "explotado",
    "expresado",
    "extendido",
    "extraviado",
    "fabricado",
    "falcado",
    "fallido",
    "fechado",
    "federado",
    "firmado",
    "fiscalizado",
    "flambeado",
    "fluido",
    "fornido",
    "fortificado",
    "forzado",
    "fracasado",
    "fragmentado",
    "frustrado",
    "fundamentado",
    "galardonado",
    "ganado",
    "garantizado",
    "gastado",
    "generalizado",
    "habilitado",
    "habitado",
    "habituado",
    "helado",
    "hendido",
    "herido",
    "hipotecado",
    "hojeado",
    "homologado",
    "honrado",
    "horrorizado",
    "hospedado",
    "humorado",
    "hundido",
    "identificado",
    "igualado",
    "iluminado",
    "ilustrado",
    "implantado",
    "implicado",
    "importado",
    "imposibilitado",
    "impostado",
    "impregnado",
    "impresionado",
    "improvisado",
    "imputado",
    "inaugurado",
    "incapacitado",
    "incluido",
    "incrementado",
    "indemnizado",
    "indeterminado",
    "indicado",
    "indignado",
    "indiscutido",
    "inesperado",
    "infectado",
    "inflamado",
    "informado",
    "informatizado",
    "insonorizado",
    "instalado",
    "integrado",
    "intencionado",
    "interconectado",
    "interesado",
    "internado",
    "interpretado",
    "inundado",
    "invalidado",
    "inventado",
    "invertido",
    "invitado",
    "involucrado",
    "irritado",
    "justificado",
    "latinizado",
    "laureado",
    "legado",
    "lesionado",
    "liberado",
    "librado",
    "licenciado",
    "ligado",
    "limitado",
    "listado",
    "llamado",
    "llevado",
    "localizado",
    "malhablado",
    "malogrado",
    "maltratado",
    "marcado",
    "marinado",
    "mecanizado",
    "megadentado",
    "mejorado",
    "mencionado",
    "mesurado",
    "metido",
    "moderado",
    "modificado",
    "montado",
    "multiplicado",
    "mutado",
    "nacionalizado",
    "navalizado",
    "nimbado",
    "nombrado",
    "nutrido",
    "obligado",
    "obsesionado",
    "ocupado",
    "ocurrido",
    "ofendido",
    "olvidado",
    "ondeado",
    "ondulado",
    "ordenado",
    "organizado",
    "ornado",
    "osado",
    "parado",
    "paralizado",
    "pasado",
    "pausado",
    "pavimentado",
    "pegado",
    "pensado",
    "pensionado",
    "permitido",
    "personalizado",
    "pervertido",
    "pesado",
    "pintado",
    "planchado",
    "planeado",
    "planificado",
    "plantado",
    "plasmado",
    "plateado",
    "plegado",
    "poblado",
    "podrido",
    "policromado",
    "polidentado",
    "ponderado",
    "pormenorizado",
    "practicado",
    "preciado",
    "preferido",
    "premeditado",
    "prensado",
    "preocupado",
    "preparado",
    "preservado",
    "presionado",
    "privado",
    "privilegiado",
    "programado",
    "prohibido",
    "prolongado",
    "promocionado",
    "pronosticado",
    "pronunciado",
    "protagonizado",
    "protegido",
    "provocado",
    "publicado",
    "quebrado",
    "quemado",
    "querido",
    "radicado",
    "rallado",
    "rapado",
    "rasurado",
    "reaprovechado",
    "recargado",
    "rechazado",
    "recibido",
    "recogido",
    "recordado",
    "recortado",
    "recuperado",
    "redondeado",
    "refinado",
    "reflejado",
    "reforzado",
    "registrado",
    "reglado",
    "reiterado",
    "relacionado",
    "relajado",
    "relatado",
    "remasterizado",
    "remezclado",
    "renombrado",
    "renovado",
    "repartido",
    "repetido",
    "reportado",
    "reputado",
    "requerido",
    "resaltado",
    "reservado",
    "respetado",
    "restringido",
    "resumido",
    "retenido",
    "reticulado",
    "retirado",
    "retrasado",
    "reñido",
    "rizado",
    "rodeado",
    "sacado",
    "sagrado",
    "salado",
    "santificado",
    "secuestrado",
    "segmentado",
    "seguido",
    "seleccionado",
    "sellado",
    "semielevado",
    "sensibilizado",
    "sentado",
    "señalado",
    "señalizado",
    "simplificado",
    "sincronizado",
    "sindicado",
    "situado",
    "sobrado",
    "sobrecargado",
    "sobreelevado",
    "sofisticado",
    "sofocado",
    "solemnizado",
    "solicitado",
    "sometido",
    "sonado",
    "soportado",
    "soterrado",
    "suavizado",
    "subcargado",
    "subordinado",
    "sujeto",
    "sumido",
    "suministrado",
    "supeditado",
    "superado",
    "suprimido",
    "tabulado",
    "tachonado",
    "talado",
    "tallado",
    "tasado",
    "techado",
    "tecnificado",
    "templado",
    "terminado",
    "tirado",
    "tocado",
    "tostado",
    "trabado",
    "trabajado",
    "transitado",
    "trasladado",
    "trastornado",
    "tratado",
    "trazado",
    "tutelado",
    "ubicado",
    "unificado",
    "uniseriado",
    "usado",
    "utilizado",
    "variado",
    "venido",
    "vestido",
    "vigilado",
    "vinculado",
    "volado",
    "votado",
    "zambullido",
    "zombificado",

    # starting here are words *not* tagged Part from GSD which probably should have been
    "abnegado",
    "ajardinado",
    "apilastrado",
    "apoderado",
    "atolondrado",
    "callado",
    "cancelado",
    "debido",
    "delegado",
    "deliberado",
    "demasiado",
    "desmesurado",
    "desprotegido",
    "entregado",
    "exaltado",
    "inflado",
    "mediado",
    "necesitado",
    "reformado",

    # these participles are from PUD but not GSD, mostly reviewed
    "acentuado",
    "afortunado",
    "agavillado",
    "apasionado",
    "carbonatado",
    "desenfocado",
    "despejado",
    "desplazado",
    "desvencijado",
    "discutido",
    "idealizado",
    "ilimitado",
    "premiado",
    "representado",

    # participles from AnCora but not PUD or GSD with some checking
    "agolpado",
    "alertado",
    "arropado",
    "clavado",
    "desconcertado",
    "encorsetado",
    "entusiasmado",
    "exhibido",
    "expedientado",
    "implicado",
    "imbuido",
    "insertado",
    "llegado",
    "preguntado",
    "presentado",
    "presidido",
    "rehabilitado",
    "rendido",
    "reconstituido",
    "reunido",
    "revestido",
    "rodado",
    "surgido",
    "tomado",
    "turbado",
    "volcado",

    # participles from AnCora which are partially checked
    "abarrotado",
    "abatido",
    "abolido",
    "abonado",
    "abrasado",
    "abrazado",
    "abrumado",
    "absorbido",
    "acabado",
    "acaecido",
    "acalambrado",
    "acancionado",
    "aceptado",
    "acertado",
    "aclarado",
    "acometido",
    "acontecido",
    "acosado",
    "acrecentado",
    "acreditado",
    "acuñado",
    "adeudado",
    "adherido",
    "adiestrado",
    "administrado",
    "admitido",
    "adoptado",
    "adorado",
    "adormecido",
    "adulterado",
    "advertido",
    "afeitado",
    "aforado",
    "agarrotado",
    "agobiado",
    "aguijoneado",
    "airado",
    "alcantarillado",
    "aleccionado",
    "alentado",
    "alicatado",
    "alimentado",
    "alineado",
    "alistado",
    "allegado",
    "alocado",
    "alquilado",
    "amalgamado",
    "amargado",
    "amedrentado",
    "amortajado",
    "analizado",
    "andado",
    "anegado",
    "angustiado",
    "anotado",
    "anudado",
    "anulado",
    "anunciado",
    "apadrinado",
    "apagado",
    "aparcado",
    "aparejado",
    "apartado",
    "apeado",
    "apelado",
    "apesadumbrado",
    "apestado",
    "aplaudido",
    "aplazado",
    "apodado",
    "aportado",
    "apoyado",
    "aprendido",
    "apresurado",
    "aprobado",
    "aquejado",
    "arañado",
    "arraigado",
    "arrojado",
    "asalariado",
    "ascendido",
    "asediado",
    "asesinado",
    "asesorado",
    "asfaltado",
    "asignado",
    "asistido",
    "asombrado",
    "astillado",
    "asumido",
    "atareado",
    "ataviado",
    "atomizado",
    "atravesado",
    "atrincherado",
    "aturullado",
    "auspiciado",
    "autoconvencido",
    "autogenerado",
    "autoproclamado",
    "autorizado",
    "avalado",
    "averiado",
    "avezado",
    "avinagrado",
    "avisado",
    "avispado",
    "ayudado",
    "azotado",
    "azucarado",
    "bajado",
    "balcanizado",
    "barajado",
    "batido",
    "bautizado",
    "bebido",
    "becado",
    "bombardeado",
    "bordeado",
    "borrado",
    "botado",
    "bregado",
    "burlado",
    "burocratizado",
    "buscado",
    "cabreado",
    "cacareado",
    "calcinado",
    "calculado",
    "calificado",
    "camuflado",
    "capitaneado",
    "captado",
    "caracterizado",
    "carbonizado",
    "castigado",
    "catalogado",
    "causado",
    "cegado",
    "censado",
    "censurado",
    "cercado",
    "cesado",
    "chafado",
    "chantajeado",
    "cifrado",
    "circulado",
    "circunstanciado",
    "clasificado",
    "colapsado",
    "colegiado",
    "colgado",
    "coloreado",
    "comandado",
    "comparado",
    "compenetrado",
    "compensado",
    "computerizado",
    "comunicado",
    "concebido",
    "concedido",
    "concienciado",
    "concretado",
    "condonado",
    "confeccionado",
    "confirmado",
    "confiscado",
    "conformado",
    "congelado",
    "congregado",
    "conjugado",
    "conjuntado",
    "conmocionado",
    "conmovido",
    "conquistado",
    "consensuado",
    "constituido",
    "consumido",
    "contemplado",
    "contestado",
    "contraindicado",
    "convertido",
    "coorganizado",
    "coronado",
    "cosechado",
    "cotizado",
    "criminalizado",
    "criticado",
    "culminado",
    "cultivado",
    "cursado",
    "curtido",
    "custodiado",
    "damnificado",
    "debilitado",
    "declarado",
    "decomisado",
    "decretado",
    "defenestrado",
    "defraudado",
    "dejado",
    "demacrado",
    "denostado",
    "denunciado",
    "depositado",
    "depreciado",
    "derramado",
    "desactivado",
    "desaforado",
    "desairado",
    "desandado",
    "desarticulado",
    "desautorizado",
    "desbaratado",
    "desbocado",
    "desbordado",
    "descabellado",
    "descalificado",
    "descansado",
    "descarado",
    "desclasificado",
    "descolgado",
    "descolorido",
    "descompensado",
    "desconfiado",
    "desconsiderado",
    "desconsolado",
    "descontado",
    "descoordinado",
    "descoyuntado",
    "descuidado",
    "desembozado",
    "desempeñado",
    "desencajado",
    "desencantado",
    "desertizado",
    "desesperanzado",
    "desestimado",
    "desfavorecido",
    "desgajado",
    "desgraciado",
    "desguazado",
    "desheredado",
    "deshidratado",
    "deshilachado",
    "deshojado",
    "deshuesado",
    "desinhibido",
    "deslocalizado",
    "desmadrado",
    "desmedido",
    "desmenuzado",
    "desmochado",
    "desmoralizado",
    "desnaturalizado",
    "desolado",
    "desorbitado",
    "desorganizado",
    "desorientado",
    "desparramado",
    "despatarrado",
    "despenalizado",
    "desperdigado",
    "despertado",
    "desplegado",
    "despojado",
    "despreocupado",
    "desprovisto",
    "destemplado",
    "destituido",
    "destripado",
    "destronado",
    "destrozado",
    "desunido",
    "desviado",
    "desvinculado",
    "detectado",
    "devaluado",
    "devastado",
    "dialogado",
    "dictado",
    "difamado",
    "diferido",
    "difuminado",
    "diligenciado",
    "diluido",
    "discriminado",
    "diseñado",
    "disociado",
    "disparado",
    "dispensado",
    "distanciado",
    "distorsionado",
    "distribuido",
    "divulgado",
    "dolido",
    "dominado",
    "donado",
    "dopado",
    "dramatizado",
    "echado",
    "editado",
    "efectuado",
    "ejecutado",
    "eliminado",
    "embadurnado",
    "embelesado",
    "embestido",
    "embrollado",
    "embrutecido",
    "emigrado",
    "emitido",
    "empadronado",
    "emparejado",
    "empedrado",
    "empingorotado",
    "empleado",
    "emprendido",
    "enajenado",
    "encabezado",
    "encadenado",
    "encajado",
    "encajonado",
    "encaminado",
    "encapuchado",
    "encarcelado",
    "encarnado",
    "encausado",
    "encerrado",
    "encharcado",
    "encontrado",
    "encuestado",
    "enfrentado",
    "enfriado",
    "enganchado",
    "engrasado",
    "enmascarado",
    "enrabietado",
    "enrocado",
    "enrollado",
    "ensalzado",
    "ensangrentado",
    "ensartado",
    "ensimismado",
    "ensombrecido",
    "entablado",
    "entendido",
    "enterado",
    "enterrado",
    "entrecortado",
    "entrenado",
    "entrevistado",
    "entristecido",
    "entrometido",
    "enumerado",
    "envasado",
    "envejecido",
    "enviado",
    "envidiado",
    "equivocado",
    "escayolado",
    "escindido",
    "escogido",
    "escoltado",
    "escrutado",
    "esgrimido",
    "esparcido",
    "especificado",
    "esquilmado",
    "estabilizado",
    "estampado",
    "estimado",
    "estipulado",
    "estrenado",
    "estreñido",
    "estructurado",
    "evolucionado",
    "exigido",
    "explorado",
    "expoliado",
    "exportado",
    "expropiado",
    "expulsado",
    "extinguido",
    "extrañado",
    "facilitado",
    "falseado",
    "falsificado",
    "familiarizado",
    "favorecido",
    "feriado",
    "fichado",
    "fijado",
    "filmado",
    "finalizado",
    "financiado",
    "fingido",
    "forjado",
    "formulado",
    "fotografiado",
    "fraccionado",
    "frenado",
    "fruncido",
    "fulminado",
    "fundado",
    "fusionado",
    "generado",
    "gestionado",
    "girado",
    "globalizado",
    "gobernado",
    "goleado",
    "golpeado",
    "grabado",
    "gratificado",
    "guiado",
    "hablado",
    "hallado",
    "heredado",
    "hidratado",
    "hidrocarbonado",
    "hinchado",
    "hipnotizado",
    "hospitalizado",
    "huido",
    "humillado",
    "ideado",
    "ignorado",
    "ilusionado",
    "imaginado",
    "imitado",
    "impedido",
    "impulsado",
    "incautado",
    "incendiado",
    "incentivado",
    "incorporado",
    "incrustado",
    "independizado",
    "indexado",
    "individualizado",
    "indultado",
    "industrializado",
    "infiltrado",
    "influido",
    "infravalorado",
    "ingresado",
    "inhabilitado",
    "iniciado",
    "inmigrado",
    "inmovilizado",
    "inmunizado",
    "insinuado",
    "inspirado",
    "instaurado",
    "instigado",
    "institucionalizado",
    "intercalado",
    "interiorizado",
    "intermediado",
    "interrumpido",
    "intervenido",
    "intoxicado",
    "inutilizado",
    "investigado",
    "inveterado",
    "invocado",
    "ionizado",
    "irisado",
    "irradiado",
    "izado",
    "jubilado",
    "jugado",
    "jurado",
    "juzgado",
    "lamido",
    "lanzado",
    "lavado",
    "levantado",
    "liberalizado",
    "licuado",
    "liderado",
    "liofilizado",
    "logrado",
    "maleado",
    "malherido",
    "manifestado",
    "manipulado",
    "manoseado",
    "mantenido",
    "maravillado",
    "marginado",
    "materializado",
    "matizado",
    "matriculado",
    "medido",
    "mezclado",
    "militarizado",
    "minado",
    "modernizado",
    "modulado",
    "mojado",
    "monocargado",
    "mordisqueado",
    "mostrado",
    "moteado",
    "motivado",
    "motorizado",
    "movido",
    "multado",
    "mutilado",
    "naufragado",
    "negado",
    "negociado",
    "nevado",
    "obcecado",
    "obsequiado",
    "observado",
    "obstinado",
    "obtenido",
    "ocasionado",
    "ocultado",
    "odiado",
    "ofertado",
    "operado",
    "orientado",
    "originado",
    "otorgado",
    "oxidado",
    "pactado",
    "pagado",
    "parapetado",
    "participado",
    "pateado",
    "patentado",
    "patrocinado",
    "pautado",
    "pedido",
    "peinado",
    "penado",
    "percibido",
    "perfeccionado",
    "perfilado",
    "perforado",
    "perjudicado",
    "perlado",
    "perpetrado",
    "perseguido",
    "pertrechado",
    "plagado",
    "planteado",
    "polarizado",
    "policonjugado",
    "politizado",
    "posicionado",
    "precavido",
    "precedido",
    "precipitado",
    "preconcebido",
    "preconizado",
    "predestinado",
    "predeterminado",
    "predispuesto",
    "prefigurado",
    "prefijado",
    "preseleccionado",
    "prestado",
    "presupuestado",
    "pretendido",
    "preñado",
    "priorizado",
    "privatizado",
    "probado",
    "procesado",
    "proclamado",
    "prologado",
    "prometido",
    "promovido",
    "promulgado",
    "propiciado",
    "propinado",
    "protestado",
    "proyectado",
    "publicitado",
    "radiocontrolado",
    "ralentizado",
    "ratificado",
    "rayado",
    "realizado",
    "rebajado",
    "rebasado",
    "rebautizado",
    "recabado",
    "recalentado",
    "reciclado",
    "reclamado",
    "recluido",
    "reclutado",
    "recomendado",
    "reconvertido",
    "recopilado",
    "recorrido",
    "redactado",
    "redoblado",
    "reelegido",
    "referido",
    "refugiado",
    "refundido",
    "refunfuñado",
    "regalado",
    "regulado",
    "relegado",
    "remitido",
    "remolcado",
    "remontado",
    "replegado",
    "resabiado",
    "rescatado",
    "resentido",
    "resignado",
    "resistido",
    "respaldado",
    "restaurado",
    "retardado",
    "retocado",
    "retribuido",
    "reunificado",
    "revenido",
    "revisado",
    "revitalizado",
    "ribeteado",
    "ridiculizado",
    "robado",
    "sabido",
    "saboreado",
    "sacudido",
    "salido",
    "salpicado",
    "saludado",
    "salvado",
    "sancionado",
    "saneado",
    "saqueado",
    "segregado",
    "sentenciado",
    "sentido",
    "sepultado",
    "servido",
    "simbolizado",
    "simulado",
    "sintetizado",
    "sobredimensionado",
    "sobresaltado",
    "sobreseído",
    "sodomizado",
    "soleado",
    "soliviantado",
    "sorprendido",
    "sostenido",
    "soñado",
    "subcontratado",
    "subido",
    "subtitulado",
    "subvencionado",
    "sucedido",
    "sufrido",
    "sugerido",
    "sumado",
    "sumergido",
    "surtido",
    "suscitado",
    "suspendido",
    "sustentado",
    "sustituido",
    "tachado",
    "tapado",
    "tejido",
    "televisado",
    "temido",
    "tendido",
    "tentado",
    "teñido",
    "tildado",
    "tipificado",
    "titulado",
    "tonsurado",
    "traducido",
    "traicionado",
    "transcurrido",
    "transformado",
    "transmitido",
    "transportado",
    "trasnochado",
    "trasplantado",
    "trenzado",
    "tronado",
    "trufado",
    "tupido",
    "ultracongelado",
    "ungido",
    "urdido",
    "valorado",
    "varado",
    "vedado",
    "vendido",
    "vertido",
    "vetado",
    "viciado",
    "vidriado",
    "vilipendiado",
    "violado",
    "visitado",
    "vivido",
    "zanjado",

    # These are words in AnCora that were not previously tagged VerbForm=Part
    # however, it looks like this collection could have that feature
    "agregado",
    "azufrado",
    "cariado",
    "desapasionado",
    "desempleado",
    "enfervorecido",
    "estrellado",
    "fidelizado",
    "granulado",
    "laminado",
    "malpensado",
    "pregrabado",
    "presumido",
    "refrigerado",
    "siniestrado",
])

# cida ending for something that kills other things: usually invariable
#  for example: genocida
# cida ending for participle, not invariable
# other exceptions exist such as ácida
cida_exceptions = [
    "ácida",

    # a long list of participles: may be better to flip it
    "conocida",
    "establecida",
    "enrojecida",
    "cocida",
    "agradecida", "aparecida", "conducida", "convencida", "coproducida", "desaparecida", "desconocida", "desvanecida", "ejercida", "empobrecida", "enaltecida", "enardecida", "enloquecida", "entorpecida", "entristecida", "envejecida", "esclarecida", "escocida", "fallecida", "introducida", "merecida", "nacida", "ofrecida", "oscurecida", "parecida", "producida", "reconocida", "reducida", "reproducida", "retorcida", "reverdecida", "vencida",
]

ista_exceptions = [
    "lista",

    "vista",
    "prevista",
    "provista",
]

def is_special_case_gender(word):
    if word in special_cases:
        return True
    if word.endswith("ista") and not word in ista_exceptions:
        return True
    if word.endswith("ísta"):
        return True
    if word.endswith("ícola"):
        return True
    if word.endswith("cida") and not word in cida_exceptions:
        return True
    if word.endswith("crata"):
        return True
    if word.endswith("arca"):
        return True
    if word.endswith("e"):
        return True
    return False

def print_inconsistent_lemmas(adjectives):
    for adj, candidates in adjectives.items():
        if len(candidates) > 1:
            print(adj, candidates)

def get_proposed_replacements(adjectives, known_only=True):
    replacements = {}
    for adj, candidates in adjectives.items():
        if len(candidates) == 2:
            candidates = sorted(candidates)
            if not candidates[0].endswith("a") or not candidates[1].endswith("o") or candidates[0][:-1] != candidates[1][:-1]:
                continue
            #print(adj, candidates)
            if known_only and candidates[1] not in adjectives and "%ss" % candidates[1] not in adjectives:
                continue
            if is_special_case_gender(candidates[0]):
                print("WARNING: check %s" % candidates[0])
                continue
            # now we have a male form which exists as an ADJ
            # but the feminine form was used as a lemma
            # we have high confidence this should be updated
            #print("  ", adj, candidates[1])
            replacements[adj] = candidates[1]
        elif len(candidates) == 1:
            candidate = next(iter(candidates))
            if not candidate.endswith("a") and not candidate.endswith("as"):
                continue
            if is_special_case_gender(candidate):
                continue
            if is_special_case_gender(adj.lower()) and candidate.endswith("o"):
                print("WARNING: check %s" % candidates[0])
                continue
            if candidate.endswith("a"):
                candidate = candidate[:-1] + "o"
            elif candidate.endswith("as"):
                candidate = candidate[:-2] + "o"
            if candidate in adjectives:
                replacements[adj] = candidate
    return replacements

def update_adjectives(adjectives, filenames, known_only=True):
    replacements = get_proposed_replacements(adjectives, known_only)
    print("Number of proposed replacements: %d" % len(replacements))
    if len(replacements) == 0:
        return
    for filename in filenames:
        print(filename)
        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != 'ADJ':
                    continue
                word.lemma = replacements.get(word.text, word.lemma)
        CoNLL.write_doc2conll(doc, filename)

def search_a_ending_adjectives(adjectives):
    printed = False
    for adj, candidates in adjectives.items():
        if not adj.endswith("a") and not adj.endswith("as"):
            continue
        if len(candidates) != 1:
            continue
        candidate = list(candidates)[0]
        if not candidate.endswith("a"):
            if is_special_case_gender(candidate):
                print("WARNING: check words with the lemma %s" % candidate)
            continue
        if is_special_case_gender(candidate):
            continue
        if not printed:
            printed = True
            print("The following -a ADJ have -a in the lemma, likely incorrectly")
        if adj.lower() == candidate or "%ss" % candidate == adj.lower():
            print(adj)
        else:
            print("%s ... %s" % (adj, candidate))
        
FILENAMES = [
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-train.conllu",
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-dev.conllu",
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-test.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-train.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-dev.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-test.conllu",
    "extern_data/ud2/git/UD_Spanish-PUD/es_pud-ud-test.conllu",
]

COSER = [
    "extern_data/ud2/git/UD_Spanish-Coser/es_coser-ud-test.conllu",
]

adj_ending = re.compile("^.*[aAoO]([sS])?$")

def load_adjectives(filenames):
    lemmas = defaultdict(set)
    features = defaultdict(set)

    for filename in filenames:
        print(filename)

        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != 'ADJ':
                    continue
                if word.feats and "Typo=Yes" in word.feats:
                    continue
                if word.feats and "Foreign=Yes" in word.feats:
                    continue
                #if adj_ending.match(word.text):
                lemmas[word.text].add(word.lemma)
                features[word.text].add(word.feats)

    print(len(lemmas))
    return lemmas, features

def show_missing_features(filenames):
    for filename in filenames:
        print(filename)
        unknown_words = set()
        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
          for word in sentence.words:
              if word.pos != 'ADJ':
                  continue
              if not word.feats:
                  if word.text in no_feature_cases:
                      continue
                  if word.text in unknown_words:
                      continue
                  unknown_words.add(word.text)
                  print(word.text)

def update_missing_verbform(filenames):
    for filename in filenames:
        changed = False
        print(filename)
        unknown_words = set()
        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
          for word in sentence.words:
              if word.pos != 'ADJ':
                  continue
              if word.deprel == 'fixed':
                  continue
              if normalize_participle(word.text) in known_participles:
                  gender = "Fem" if word.text.lower().endswith("a") or word.text.lower().endswith("as") else "Masc"
                  number = "Plur" if word.text.lower().endswith("s") else "Sing"
                  feats = "Gender=%s|Number=%s|VerbForm=Part" % (gender, number)
                  if not word.feats or ('ExtPos' not in word.feats and 'Typo' not in word.feats):
                      changed = True
                      word.feats = feats

        if changed:
            CoNLL.write_doc2conll(doc, filename)

def show_spurious_features(filenames, execute=True):
    """
    Looks at the individual words looking for ADJ which have features we know shouldn't exist

    for example: if the word is not expected to have features but does
    or if it doesn't inflect for Gender but has a Gender feature
    """
    unwanted_features = set()
    for filename in filenames:
        print(filename)
        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != 'ADJ':
                    continue
                if not word.feats:
                    continue
                if word.lemma in no_feature_cases:
                    print("Features on %s: %s" % (word.lemma, word.feats))
                    unwanted_features.add(word.lemma)
                elif is_special_case_gender(word.text):
                    pieces = word.feats.split("|")
                    if any(x.startswith("Gender") for x in pieces):
                        print("Features w/ gender on %s: %s" % (word.lemma, word.feats))
                        unwanted_features.add(word.lemma)
                        if execute:
                            feats = "|".join([x for x in pieces if not x.startswith("Gender")])
                            word.feats = feats
        if execute:
            CoNLL.write_doc2conll(doc, filename)

    for lemma in sorted(unwanted_features):
        print(lemma)

def check_special_cases(adjectives):
    """
    Look through the words invariable for Gender and report any which had a -o lemma anyway
    """
    potential_problems = set()
    for adj, candidates in adjectives.items():
        if is_special_case_gender(adj):
            if len(candidates) != 1:
                print("Weird candidates list for %s: %s" % (adj, candidates))
            candidate = next(iter(candidates))
            if not adj.endswith("o") and candidate.endswith("o"):
                print("Special case ADJ with probably incorrect lemma: %s -> %s" % (adj, candidate))
                potential_problems.add(adj)
    for adj in sorted(potential_problems):
        print(adj)

def normalize_participle(adj):
    adj = adj.lower()
    if adj.endswith("s"):
        adj = adj[:-1]
    if adj.endswith("a"):
        adj = adj[:-1] + "o"
    return adj

def check_verbform_features(features, filenames):
    """
    Report any words with VerbForm=Inf or unknown VerbForm=Part
    """
    infinitives = set()
    participles = set()
    other = set()
    for adj, features in features.items():
        for feat in features:
            if not feat:
                continue
            pieces = feat.split("|")
            pieces = [x for x in pieces if x.startswith("VerbForm")]
            if any(x == "VerbForm=Inf" for x in pieces):
                infinitives.add(adj)
            elif any(x == "VerbForm=Part" for x in pieces):
                if normalize_participle(adj) not in known_participles:
                    participles.add(adj)
            elif any(x.startswith("VerbForm") for x in pieces):
                other.add(adj)
    if len(infinitives) > 0:
        print("ADJ with marked VerbForm=Inf:")
        for x in sorted(infinitives):
            print(x)
    if len(other) > 0:
        print("ADJ with marked VerbForm=???:")
        for x in sorted(other):
            print(x)
    if len(participles) > 0:
        print("ADJ with marked VerbForm=Part:")
        for x in sorted(participles):
            print(x)

def check_unlabeled_do_verbforms(features):
    print("Checking for potential uncaught participles...")
    unknown_adj = set()
    for adj, feat in features.items():
        adj = normalize_participle(adj)
        if not adj.endswith("ado") and not adj.endswith("ido"):
            continue
        if adj in known_participles or adj in known_non_participles:
            continue
        if adj[:-1] + "a" in known_non_participles:
            continue
        if any(x is not None and "VerbForm=Part" in x for x in feat):
            continue
        unknown_adj.add(adj)
    for adj in sorted(unknown_adj):
        print(adj)


def main():
    for filename in FILENAMES:
        assert os.path.exists(filename)
    for filename in COSER:
        assert os.path.exists(filename)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['pud', 'gsd', 'ancora', 'coser'], help='Only use this dataset')
    args = parser.parse_args()

    if args.dataset == 'coser':
        filenames = COSER
    elif args.dataset:
        filenames = [x for x in FILENAMES if args.dataset in x]
    else:
        filenames = FILENAMES

    lemmas, features = load_adjectives(filenames)
    print_inconsistent_lemmas(lemmas)
    update_adjectives(lemmas, filenames, known_only=False)
    search_a_ending_adjectives(lemmas)
    check_special_cases(lemmas)
    #show_spurious_features(filenames)

    #show_missing_features(filenames)

    #check_verbform_features(features, filenames)
    update_missing_verbform(filenames)
    #check_unlabeled_do_verbforms(features)

if __name__ == '__main__':
    main()

