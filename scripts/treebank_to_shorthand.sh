declare -A lang2lcode=( ["Afrikaans"]="af" ["Armenian"]="hy" ["Arabic"]="ar" ["Breton"]="br" ["Bulgarian"]="bg" ["Catalan"]="ca" ["Czech"]="cs" ["Old_Church_Slavonic"]="cu" ["Danish"]="da" ["German"]="de" ["Greek"]="el" ["English"]="en" ["Spanish"]="es" ["Estonian"]="et" ["Basque"]="eu" ["Persian"]="fa" ["Finnish"]="fi" ["French"]="fr" ["Irish"]="ga" ["Galician"]="gl" ["Gothic"]="got" ["Ancient_Greek"]="grc" ["Hebrew"]="he" ["Hindi"]="hi" ["Croatian"]="hr" ["Hungarian"]="hu" ["Indonesian"]="id" ["Italian"]="it" ["Japanese"]="ja" ["Kazakh"]="kk" ["Korean"]="ko" ["Latin"]="la" ["Latvian"]="lv" ["Dutch"]="nl" ["Norwegian"]="no" ["Polish"]="pl" ["Portuguese"]="pt" ["Romanian"]="ro" ["Russian"]="ru" ["Slovak"]="sk" ["Slovenian"]="sl" ["Swedish"]="sv" ["Turkish"]="tr" ["Uyghur"]="ug" ["Ukrainian"]="uk" ["Urdu"]="ur" ["Vietnamese"]="vi" ["Chinese"]="zh" ["Altaic"]="bxr" ["Indo_Iranian"]="kmr" ["Uralic"]="sme" ["Slavic"]="hsb" )

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
