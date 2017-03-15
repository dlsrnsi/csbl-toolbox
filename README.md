# ID converter

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

# To do
- [x] add option '--replace' to determine coverted id will replace original id
- [ ] add option `--sep` to specify seperator 


