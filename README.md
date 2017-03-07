# ID converter

## Dependencies

	- `R > 3.2.5`
	- 'python > 2.7'
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
## help

```
Rscript ID_converter.R --help
```

## example
read from csv and convert from ensembl gene id to hgnc symbol and replace original file
```
Rscript ID_converter --inplace --header ensembl_gene_id hgnc_symbol ./file_path
```

### things to do

add option `--replace` for replacing original column(not recommended)