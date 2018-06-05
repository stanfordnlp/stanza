args=$@

n_gpus=`gpustat | tail -n+2 | wc -l`

i=0
for tb in UD_Chinese-GSD UD_English-EWT UD_French-Spoken UD_Hebrew-HTB UD_Japanese-GSD UD_Old_Church_Slavonic-PROIEL UD_Vietnamese-VTB; do
    ./run.sh $tb $((i % n_gpus)) $args &
    pids[${i}]=$!
    i=$((i+1))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

for t in zh_gsd en_ewt fr_spoken he_htb ja_gsd cu_proiel vi_vtb; do
    echo `cat data/${t}.results | grep -v '^--' | awk '{if ($1 == 0 || $2 == 0 || $3 == 0) print 0; else print (3/(1/$1+1/$2+1/$3))}' | tail -1 `;
done | awk '{total = total + $1}END{print total/7}' >&2
