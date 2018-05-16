#!/bin/bash

if [ $# -lt 2 ]
then
    echo "Usage: $0 <Balls/Goals> <zip_file>"
    exit -1
fi

# Reading arguments
scripts_path=$(dirname $0 )
feature_name=$1
zip_file=$2

img_path=Images/${feature_name}
pos_path=${img_path}/positive/
neg_path=${img_path}/negative/

# Extract data to given folder
rm -rf $img_path
mkdir -p ${img_path}
unzip $zip_file -d ${img_path}
# Extracting positive images
mkdir $pos_path
python ${scripts_path}/get_from_json.py ${img_path}/data.json
# Moving negative images
mkdir $neg_path
mv ${img_path}/*.png ${neg_path}
# Adding noise
python ${scripts_path}/add_noise.py ${pos_path}
python ${scripts_path}/add_noise.py ${neg_path}
