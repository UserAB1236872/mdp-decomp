python main_scaii.py --train --total_episodes 10000 --restore
i="0"
while [ $? -eq 0 ]; do
    python main_scaii.py --train --total_episodes 10000 --restore
    i=$[$i+1]

    if [ $i -gt 4 ]; then
        echo "Reached max attempts"
        exit
    fi   
done