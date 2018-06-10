args=$@

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
tbs=`wc -l tokenize_test_treebanks`
for tb in `cat tokenize_test_treebanks`; do
    sbatch --wait $ROOT/scripts/run_slurm_tokenize.sh $tb $ROOT $args 2>/dev/null &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in `cat tokenize_test_treebanks | xargs -i bash scripts/treebank_to_shorthand.sh ud {}`; do
    echo `cat data/tokenize/${t}.results | grep -v '^--' | awk '{if ($1 == 0 || $2 == 0) print 0; else print (2/(1/$1+1/$2))}' | tail -1 `;
done | awk '{total = total + $1; count = count + 1}END{print total/count}' >&2
