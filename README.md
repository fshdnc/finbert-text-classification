# finbert-text-classification
Text classification experiments with BERT

## Repo setup
ylilauta-corpus -> ../../../project_2002085/sampo/finbert-text-classification/ylilauta-corpus
yle-corpus -> ../../../project_2001553/sampo/yle-corpus
bert -> ../../../project_2002085/lihsin/bert

1. Gridding
The following parameters are not searched, but directly chosen based on FinBERT experiments
lr {2e-5} # FinBERT gridding result 66 out of 80
epoch {4] # FinBERT gridding result 37 out of 80
Use these two scripts to do gridding `batch-run-parameter-selection.sh` and `batch-run-dev.sh`.

2. Select parameters
`grep -h DEV-RESULT output/*.out | grep eval_accuracy > delme`
`python3 /scratch/project_2002820/lihsin/bert-pos/slurm/select_params.py delme > selected-params.tsv`
`perl -pi -e 's/\/scratch\/project_2002820\/lihsin\/bert_checkpoints\/jointvoc-(\d0)k\/model\.ckpt-1000000/biBERT$1/' selected-params.tsv`

3. Run selected parameters
Use `batch-run-selected-params.sh` and `batch-run-test.sh`

4. Organize results
Modifications made to `summarize.py` to allow for batch reading results. But the result file has to be generated with the following command
`grep -h 'TEST-RESULT' output/2325*.out | perl -pe 's/([\d\.]+%)$/accuracy\t$1/' > delme`

5. Draw graphs
`python3 draw.py results-all.tsv dummy`
`scp <username>@puhti.csc.fi:/scratch/project_2002820/lihsin/finbert-text-classification/dummy*.png .`