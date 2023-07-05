#!/bin/bash

# only effective on systems with conda installed
eval "$(conda shell.bash hook)"
conda activate sketchy
ctime=`date '+%Y_%m_%d_%H_%M_%S'`;

# for data in "sketchy_fall2019_peekvotes_fmt_final" "sketchy_dec6_dec13_peeks_updated"; do
for data in "sketchy_fall2019_peekvotes_fmt_final"; do
for weigh in false true; do
for eq3 in false; do
rm -f "$data$weigh$eq3.json"
cat <<EOF >> "$data$weigh$eq3.json"
{
"dataset_file": "data/$data.tsv",
"weight_user_votes": $weigh,
"skip_singleton_votes": true,
"xgboost_params": {"n_estimators": 500, "max_depth": 10},
"force_preprocessing": true,
"feature_selection": true,
"eq3": $eq3
}
EOF
python code/transform.py "data/$data.csv"
for i in {0..10..1}; do
python code/trainf.py --config "$data$weigh$eq3.json" --files "data/$data.ndjson" > "output/$data$weigh$eq3$ctime.out" 2> "output/$data$weigh$eq3$ctime.err"
done
done
done
done

# for data in "sketchy_fall2019_peekvotes_fmt_final" "sketchy_dec6_dec13_peeks_updated"; do
for data in "sketchy_fall2019_peekvotes_fmt_final"; do
for weigh in true; do
for eq3 in true; do
rm -f "$data$weigh$eq3.json"
cat <<EOF >> "$data$weigh$eq3.json"
{
"dataset_file": "data/$data.tsv",
"weight_user_votes": $weigh,
"skip_singleton_votes": true,
"xgboost_params": {"n_estimators": 500, "max_depth": 10},
"force_preprocessing": true,
"feature_selection": true,
"eq3": $eq3
}
EOF
python code/transform.py "data/$data.csv"
for i in {0..10..1}; do
python code/trainf.py --config "$data$weigh$eq3.json" --files "data/$data.ndjson" > "output/$data$weigh$eq3$ctime.out" 2> "output/$data$weigh$eq3$ctime.err"
done
done
done
done