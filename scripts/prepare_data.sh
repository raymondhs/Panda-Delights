for m in train test sample; do
    python get_ids.py $m
    python filter_data.py $m
done

for m in projects essays; do
    cat ../data/"$m"_sample_01.csv ../data/"$m"_test.csv > ../data/"$m"_sample+test.csv
done
