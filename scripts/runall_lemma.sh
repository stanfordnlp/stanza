args=$@

n_gpus=`gpustat | tail -n+2 | wc -l`
treebank_file=./lemma_test_treebanks

i=0
for tb in `cat $treebank_file`; do
    echo "Treebank: $tb"
    short=`bash scripts/treebank_to_shorthand.sh ud $tb`
    ./scripts/run_lemma.sh $tb $((i % n_gpus)) $args 2>&1 | tee logs/${short}.log &
    pids[${i}]=$!
    i=$((i+1))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo ""
echo "Training done. Printing results..."
for tb in `cat $treebank_file`; do
    short=`bash scripts/treebank_to_shorthand.sh ud $tb`
    tail -n1 logs/${short}.log
done

