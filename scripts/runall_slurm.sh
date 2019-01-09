outputprefix=$1
if [[ "$outputprefix" == "tokenize" || "$outputprefix" == "mwt" || "$outputprefix" == "lemma" || "$outputprefix" == "pos" || "$outputprefix" == "depparse" ]]; then
    outputprefix=""
else
    shift
fi
module=$1
shift
args=$@

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
tbs=`wc -l ${module}_test_treebanks`
i=0
for tb in `cat ${module}_test_treebanks`; do
    short=`bash $ROOT/scripts/treebank_to_shorthand.sh ud $tb`
    if [[ "$outputprefix" == "" ]]; then
        sbatch --wait --job-name ${module}.${short} $ROOT/scripts/run_slurm.sh $module run $tb $ROOT $args 2>/dev/null &
    else
        sbatch --wait --job-name ${module}.${short} $ROOT/scripts/run_slurm.sh $outputprefix $module run $tb $ROOT $args 2>/dev/null &
    fi
    pids[${i}]=$!
    i=$((i+1))
    sleep .1
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in `cat ${module}_test_treebanks | xargs -i bash scripts/treebank_to_shorthand.sh ud {}`; do
    if [[ $module = "tokenize" ]]; then
        echo `cat data/tokenize/${t}.slurm_results | grep -v '^--' | awk '{if ($1 == 0 || $2 == 0 || $3 == 0) print 0; else print (2.01/(1/$1+1/$2+.01/$3))}' | tail -1 `;
    else
        echo `cat data/${module}/${t}.slurm_results | grep -v '^--' | awk '{print $1}' | tail -1 `;
    fi
done | awk '{total = total + $1; count = count + 1}END{print total/count}' >&2
