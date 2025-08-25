# Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling

## Install dependencies
Python 3.12 or later.
```console
❱❱❱ pip install -r requirements.txt
```

## Prepare dataset
The experiments are performed on [MultiWOZ 2.4](https://github.com/smartyfh/MultiWOZ2.4) and [Snips](https://github.com/snipsco/snips-nlu). We provide the necessary files for both datasets in mwz2.4.zip and snips.zip files for convenience. Unzip the folders before running the scripts. Run the following script to create the required metadata for the Snips dataset.

```console
❱❱❱ python create_snips_info.py
```

## Train DST
1. Base Model
```console
❱❱❱ python train_dst.py -path=<model_path> -data=<mwz/snips>
```
2. Accountability Model (with slot classification)
```console
❱❱❱ python train_dst.py -path=<model_path> -data=<mwz/snips> -slot
```
Checkpoint for the best model: [sdey15/accountability-model-for-dst](https://huggingface.co/sdey15/accountability-model-for-dst)

## Generate DST
1. Base Model
```console
❱❱❱ python generate_dst.py -path=<model_path> -data=<mwz/snips> -best=<model_checkpoint>
```
2. Accountability Model
```console
❱❱❱ python generate_dst.py -path=<model_path> -data=<mwz/snips> -best=<model_checkpoint> -slot
```
Use -dev to generate for the dev data. Default is test data.

## Evaluate DST
1. Base Model
```console
❱❱❱ python evaluate_dst.py -path=<model_path> -data=<mwz/snips> -best=<model_checkpoint>
```
2. Accountability Model (Self-Correct)
```console
❱❱❱ python evaluate_dst.py -path=<model_path> -data=<mwz/snips> -best=<model_checkpoint> -slot
```

3. Accountability Model (Oracle-Correct)
```console
❱❱❱ python evaluate_dst.py -path=<model_path> -data=<mwz/snips> -best=<model_checkpoint> -slot -oracle
```
The results are saved in dst_*.txt files inside the model directory. Use -dev to evaluate for the dev data. The default is test data.
