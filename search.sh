# semi-text-c

for ur in 0.05 0.1 0.15 0.2 0.25
do
    for er in 0.1 0.2 0.3 0.4 0.5
    do
        for tn in 0 2
        do
            python main.py -d semi-text-c -k 0.05 -st -dd 8 -ur $ur -er $er --seed 2022 -tn $tn
        done
    done
done