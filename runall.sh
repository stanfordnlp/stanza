args=$@

n_gpus=`gpustat | tail -n+2 | wc -l`

i=0
tbs=`wc -l test_treebanks`
for tb in `cat test_treebanks`; do
    ./run.sh $tb $((i % n_gpus)) $args &
    pids[${i}]=$!
    i=$((i+1))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in `cat test_treebanks | xargs -i bash scripts/treebank_to_shorthand.sh ud {}`; do
    echo `cat data/${t}.results | grep -v '^--' | awk '{if ($1 == 0 || $2 == 0 || $3 == 0) print 0; else print (3/(1/$1+1/$2+1/$3))}' | tail -1 `;
done | awk '{total = total + $1; count = count + 1}END{print total/count}' >&2
