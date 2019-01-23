#!/bin/bash
#
# Convert a UD Treebank full name to its short code (e.g. UD_English-EWT to en_ewt). Run as:
#   ./treebank_to_shorthand.sh FORMAT TREEBANK
# where FORMAT is either ud or udpipe, and TREEBANK is the full name.

declare -A lang2lcode=( ["Afrikaans"]="af" ["Armenian"]="hy" ["Arabic"]="ar" ["Breton"]="br" ["Bulgarian"]="bg" ["Buryat"]="bxr" ["Catalan"]="ca" ["Czech"]="cs" ["Old_Church_Slavonic"]="cu" ["Danish"]="da" ["German"]="de" ["Greek"]="el" ["English"]="en" ["Spanish"]="es" ["Estonian"]="et" ["Basque"]="eu" ["Persian"]="fa" ["Faroese"]="fo" ["Finnish"]="fi" ["French"]="fr" ["Irish"]="ga" ["Galician"]="gl" ["Gothic"]="got" ["Ancient_Greek"]="grc" ["Hebrew"]="he" ["Hindi"]="hi" ["Croatian"]="hr" ["Hungarian"]="hu" ["Indonesian"]="id" ["Italian"]="it" ["Japanese"]="ja" ["Kazakh"]="kk" ["Korean"]="ko" ["Kurmanji"]="kmr" ["Latin"]="la" ["Latvian"]="lv" ["Dutch"]="nl" ["Norwegian"]="no" ["Polish"]="pl" ["Portuguese"]="pt" ["Romanian"]="ro" ["Russian"]="ru" ["Slovak"]="sk" ["Slovenian"]="sl" ["Swedish"]="sv" ["Turkish"]="tr" ["Uyghur"]="ug" ["Ukrainian"]="uk" ["Urdu"]="ur" ["Vietnamese"]="vi" ["Chinese"]="zh" ["Altaic"]="bxr" ["Indo_Iranian"]="kmr" ["Uralic"]="sme" ["Slavic"]="hsb" ["Naija"]="pcm" ["North_Sami"]="sme" ["Old_French"]="fro" ["Serbian"]="sr" ["Thai"]="th" ["Upper_Sorbian"]="hsb" )

format=$1
shift
treebank=$1
tbname=`echo $treebank | sed -e 's#^.*-##g' | tr [:upper:] [:lower:]`
lang=`echo $treebank | sed -e 's#-.*$##g' -e 's#^[^_]*_##g'`
lcode=${lang2lcode[$lang]}
if [ $format == 'udpipe' ]; then
    echo `echo $lang | tr [:upper:] [:lower:]`-${tbname}
else
    echo ${lcode}_${tbname}
fi
