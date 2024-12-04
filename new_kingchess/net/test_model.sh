#!/bin/bash

directory="./model/resnet5addexpert_11_7"



result_file_path="./result_resnet_add_expert.txt"

if [ ! -f "$result_file_path" ]; then
	touch "$result_file_path"
fi


for file in "$directory"/*
do
	    if [ -f "$file" ]; then
		    if [[ "$file" == *.pth ]]; then
		            echo "Processing file: $file"
			    python transfer_trt_resnet_test_model.py --model "$file"
			    model_output=$(python test_current_model_resnet.py)
			    echo "$file: $model_output" >> "$result_file_path"
		    fi
	    fi
done
