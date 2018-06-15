module=$1
shift
args=$@

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
tbs=`wc -l ${module}_test_treebanks`
i=0
for tb in `cat ${module}_test_treebanks`; do
    sbatch --wait --job-name $module $ROOT/scripts/run_slurm.sh $module run $tb $ROOT $args 2>/dev/null &
    pids[${i}]=$!
    i=$((i+1))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in `cat ${module}_test_treebanks | xargs -i bash scripts/treebank_to_shorthand.sh ud {}`; do
    if [[ $module = "tokenize" ]]; then
        echo `cat data/tokenize/${t}.results | grep -v '^--' | awk '{if ($2 == 0 || $3 == 0) print 0; else print (2/(1/$2+1/$3))}' | tail -1 `;
    else
        echo `cat data/${module}/${t}.results | grep -v '^--' | awk '{print $1}' | tail -1 `;
    fi
done | awk '{total = total + $1; count = count + 1}END{print total/count}' >&2
