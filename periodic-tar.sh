#!/usr/bin/env bash

csv_files=$(find clf_outputs/ -maxdepth 1 -name '*.csv' | wc -l)

while [ "$csv_files" -gt "900" ]
do  
    random_filename="results-$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32).tar.gz"

    find clf_outputs/ -maxdepth 1 -name '*.csv'  -printf "%f\n" | head -1000 | xargs tar -C "clf_outputs/" -czf "$random_filename" --remove-files
    sleep 1    
    csv_files=$(find clf_outputs/ -maxdepth 1 -name '*.csv' | wc -l)
done

