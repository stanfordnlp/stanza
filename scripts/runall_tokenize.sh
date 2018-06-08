args=$@

n_gpus=`gpustat | tail -n+2 | wc -l`

i=0
free_gpus=""
for free in `gpustat | tail -n+2 | cut -d"|" -f3 | awk '{print $3-$1}'`; do
    if [ $free -ge 3000 ]; then
        free_gpus=$free_gpus","$i
    fi
    i=$((i+1))
done

i=0
tbs=`wc -l tokenize_test_treebanks`
for tb in `cat tokenize_test_treebanks`; do
    scripts/run_tokenize.sh $tb $((i % n_gpus)) $args &
    pids[${i}]=$!
    i=$((i+1))
    while [[ $free_gpus != *"$((i % n_gpus))"* ]]; do
        i=$((i+1))
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in `cat tokenize_test_treebanks | xargs -i bash scripts/treebank_to_shorthand.sh ud {}`; do
    echo `cat data/tokenize/${t}.results | grep -v '^--' | awk '{if ($1 == 0 || $2 == 0) print 0; else print (2/(1/$1+1/$2))}' | tail -1 `;
done | awk '{total = total + $1; count = count + 1}END{print total/count}' >&2
