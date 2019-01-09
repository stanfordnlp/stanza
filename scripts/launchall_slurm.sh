module=$1
shift
mode=$1
shift
args=$@

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
for tb in `cat all_treebanks`; do
    sbatch --job-name $module $ROOT/scripts/run_slurm.sh $module $mode $tb $ROOT $args 2>/dev/null &
    sleep .1
done
