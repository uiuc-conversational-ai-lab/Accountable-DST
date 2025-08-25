# Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling

### Paper Link: [Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](https://aclanthology.org/2025.acl-long.1399/), ACL-Mains 2025

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

## Citation
```console
@inproceedings{dey-etal-2025-know,
    title = "Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational {AI} Through Accountability Modeling",
    author = {Dey, Suvodip  and
      Sun, Yi-Jyun  and
      Tur, Gokhan  and
      Hakkani-T{\"u}r, Dilek},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1399/",
    doi = "10.18653/v1/2025.acl-long.1399",
    pages = "28830--28843",
    ISBN = "979-8-89176-251-0"}
```
