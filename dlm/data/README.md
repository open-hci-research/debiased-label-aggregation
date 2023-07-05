Sketchy model training
===

## Preliminaries

A pipenv environment is highly recommended. 
I provide a `Pipfile.lock` file so that everybody works with the same software versions:
```
pipenv install
```

## Data preprocessing

Shaun provided the `sketchy_dec6_dec13_peeks_updated.csv` file so I put that file in the `data` dir and ran the following command:
```
pipenv run python code/transform.py data/sketchy_dec6_dec13_peeks_updated.csv
```

The program above will generate two files in the `data` folder:

1. `sketchy_dec6_dec13_peeks_updated.tsv` aka the groundtruth labels.
2. `sketchy_dec6_dec13_peeks_updated.ndjson` aka the sketches database.

## Model training 

I created the `sketchy_dec6_dec13_peeks_updated_conf.json` config file (already provided in the repo) and ran the following command:
```
pipenv run python code/trainf.py --config sketchy_dec6_dec13_peeks_updated_conf.json --files data/sketchy_dec6_dec13_peeks_updated.ndjson > sketchy_dec6_dec13_peeks_updated.out 2> sketchy_dec6_dec13_peeks_updated.err
```

Notice that we redirect the output (and errors) of this process to a text file, so that we review it later.

After training, several files will be created:

- `sketchy_dec6_dec13_peeks_updated-labels.csv` aka groundtruth labels.
- `sketchy_dec6_dec13_peeks_updated-samples_whiten.csv` aka _normalized_ feature vectors.
- `sketchy_dec6_dec13_peeks_updated-samples.csv` aka _raw_ feature vectors.
- `sketchy_dec6_dec13_peeks_updated-columns.csv` aka column names.
- `sketchy_dec6_dec13_peeks_updated-fimps*.csv` aka feature selection results.

These files will be reused if you run the experiments again, unless you set `force_preprocessing` to `true` in the experiment config JSON file.

## Model deployment

After model training, a `.xgb` file will be created in the root dir. 
You have to copy this file into the `inspire` nodejs dir.
**Note:** The nodejs module will load the latest model file automatically, so you don't need to rename the file.

Also **double-check the output of the tranining process**, especially if feature selection has been applied, 
since we have to indicate in `inspire/index.js` which features should be used in production. 
