mkdir -p ../data/{filtered-train,test,sample}

for m in train test sample; do
    python get_ids.py $m
    python filter_data.py $m
done

mkdir -p ../data/sample+test
for m in projects essays; do
    cat ../data/sample/"$m".csv > ../data/sample+test/"$m".csv
    tail -n +2 ../data/test/"$m".csv >> ../data/sample+test/"$m".csv
done
