source scripts/config.sh

# process args
outputprefix=$1

if [[ "$outputprefix" == "UD_"* ]]; then
    outputprefix=""
else
    shift
fi

treebank=$1
shift
task=$1
shift
dataset=$1
shift
gpu=$1
shift 

short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`
args=$@

if [[ "$args" == *"--save_dir"* ]]; then
    savedir=""
else
    savedir="--save_dir ${outputprefix}saved_models/tokenize"
fi

# set up data
DATADIR=data/${task}

if [ $task == tokenize ]; then
    txt_file=${DATADIR}/${short}.${dataset}.txt
    conll_file=${DATADIR}/${short}.${dataset}.${outputprefix}pred.conllu
    gold_file=${DATADIR}/${short}.${dataset}.gold.conllu
    mwt_json_file=${DATADIR}/${short}-ud-${dataset}-mwt.json 
    # ensure input and gold data are present
    if [ ! -e $txt_file ]; then
        cp $UDBASE/$treebank/${short}.${dataset}.txt $txt_file
    fi

    if [ ! -e $gold_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $gold_file
    fi
elif [ $task == lemma ]; then
    eval_file=${DATADIR}/${short}.${dataset}.in.conllu
    output_file=${DATADIR}/${short}.${dataset}.${outputprefix}pred.conllu
    gold_file=${DATADIR}/${short}.${dataset}.gold.conllu
    # ensure input and gold data are present
    if [ ! -e $eval_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $eval_file
    fi

    if [ ! -e $gold_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $gold_file
    fi
elif [ $task == pos ]; then
    eval_file=${DATADIR}/${short}.${dataset}.in.conllu
    output_file=${DATADIR}/${short}.${dataset}.${outputprefix}pred.conllu
    gold_file=${DATADIR}/${short}.${dataset}.gold.conllu
    # ensure input and gold data are present
    if [ ! -e $eval_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $eval_file
    fi

    if [ ! -e $gold_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $gold_file
    fi
elif [ $task == depparse ]; then
    eval_file=${DATADIR}/${short}.${dataset}.in.conllu
    output_file=${DATADIR}/${short}.${dataset}.${outputprefix}pred.conllu
    gold_file=${DATADIR}/${short}.${dataset}.gold.conllu
    # ensure input and gold data are present
    if [ ! -e $eval_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $eval_file
    fi

    if [ ! -e $gold_file ]; then
        cp $UDBASE/$treebank/${short}-ud-${dataset}.conllu $gold_file
    fi
fi

# run models

declare -A task2module=( ["lemma"]="lemmatizer" ["mwt"]="mwt_expander" ["depparse"]="parser" ["pos"]="tagger" ["tokenize"]="tokenizer" )
module=${task2module[$task]}

if [ $task == tokenize ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m models.${module} --mode predict --txt_file $txt_file --lang $lang --conll_file $conll_file --shorthand $short \
         --mwt_json_file $mwt_json_file $savedir $args
    results=`python utils/conll18_ud_eval.py -v $gold_file data/tokenize/${short}.${dataset}.${outputprefix}pred.conllu | head -5 | tail -n+3 | awk '{print $7}' | pr --columns 3 -aJT`
elif [ $task == lemma ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m models.lemmatizer --data_dir $DATADIR --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $short --mode predict
    results=`python utils/conll18_ud_eval.py -v $gold_file $output_file | grep "Lemmas" | awk '{print $7}'`
elif [ $task == pos ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m models.${module} --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict --save_dir ${outputprefix}saved_models/${task} $args
    results=`python utils/conll18_ud_eval.py -v $gold_file $output_file | head -9 | tail -n+9 | awk '{print $7}'`
elif [ $task == depparse ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -m models.${module} --eval_file $eval_file \
        --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict --save_dir ${outputprefix}saved_models/${task} $args
    results=`python utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`
fi

# display results
echo $short $results $args
