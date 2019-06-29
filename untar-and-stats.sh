#!/usr/bin/env bash


tmp_dir=$(mktemp -d)
echo $tmp_dir

tar -xzf pv056-json-clf-configs.tar.gz -C $tmp_dir 

for f in results-*.tar.gz
do
    echo $f
    tar -xzf $f -C $tmp_dir
    pv056-statistics -r $tmp_dir --raw >> test-statistics.csv
    rm $tmp_dir/*.csv
done

rm $tmp_dir/*.json
rmdir $tmp_dir
