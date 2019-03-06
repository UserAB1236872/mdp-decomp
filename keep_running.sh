python main_scaii.py --train --total_episodes 7500 --restore --eval_episodes 300
i="0"
while [ $? -eq 0 ]; do
    python main_scaii.py --train --total_episodes 7500 --restore --eval_episodes 300
    i=$[$i+1]

    if [ $i -gt 10 ]; then
        echo "Reached max attempts"
        exit
    fi   
done