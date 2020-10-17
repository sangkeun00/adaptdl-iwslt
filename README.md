# 11747_hw3
Transformer re-implementation from scratch (PyTorch)


## Prepare
To test our model on your machine, please first set up the conda environment with the provided `environment.yml` file, and activate it.

```bash
conda env update -f environment.yml
conda activate nn4nlp-hw3
```

We train/test our model on the IWSLT en-de dataset, as it allows fast experiments due to its small dataset size.
To download and preprocess the IWSLT en-de dataset, please run the following script.

```bash
./scripts/data_iwslt.sh
```

## Training

You can reproduce the baseline (i.e., no AdaScale) and AdaScale results by running the following script:
```bash
./scripts/baseline.sh
./scripts/adascale.sh
```

## Testing

Currently, we average the weights from the final 10 epochs, and evalute the BLEU score using the fairseq library.
Testing can be simply done by running the following script:

```bash
./scripts/test.sh
```
