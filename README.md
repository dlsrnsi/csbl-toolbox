# ID converter

ID converter with R bioconductor package biomaRt

## Dependencies

- `R > 3.2.5`
- `python > 2.7`
- json module in R
- json, simplejson module in python
- biomaRt in biocLite
```
source("https://bioconductor.org/biocLite.R")
biocLite("biomaRt")
```
- argparse module in R
```
install.packages("argparse")
```

## Supported gene type

"entrezgene", "hgnc_symbol", "ensembl_gene_id", "unigene", "agilentprobe", "affy_hg_u133_plus", "refseq_dna"

## Help

```
Rscript ID_converter.R --help
```

## example
read from csv and convert from ensembl gene id to hgnc symbol and replace original file
```
Rscript ID_converter --inplace --header ensembl_gene_id hgnc_symbol ./file_path.csv
```

## To do
- [x] add option '--replace' to determine coverted id will replace original id
- [ ] add option `--sep` to specify seperator 


# gramizer.py

make n-gram from protein sequence with input n

## dependency

- `pandas`

## Help

```
python gramizer.py --help
```

# prot2vec.py

protein to vector, with tensorflow module.

## dependency

- `tensorflow`
- `gramizer.py`
- `pandas`
- `numpy`

## usage as class

```
from prot2vec import Prot2vec
prot2vec =  Prot2vec(skip_window, num_skips, batch_size, embedding_size)
count, dictionary, reverse_dictionary = prot2vec.build_dataset_from_seqlist(seq_list)
final_embeddings =  prot2vec.learn()
```

## Help

```
python prot2vec.py
```

## To do

- [ ] determine how output file will saved if only sequence is targeted
- [ ] make docstring


