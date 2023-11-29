#!/bin/bash

eval "$(cat ~/.bashrc | tail -n +10)"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi


folder_path="$1"

if [ ! -d "$folder_path" ]; then
    echo "Invalid folder path. Exiting."
    exit 1
fi

find "$folder_path" -type f -name "*.ORF" | while read -r file; do

    echo "$file"
    echo "$file" > demo/file.txt
    poetry run jupyter nbconvert --to notebook --inplace --execute demo/pipeline.ipynb

done

echo "Finished"
