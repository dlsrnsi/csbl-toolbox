library(argparse)
library(biomaRt)
library(tools)
ensembl = useMart("ensembl",dataset="hsapiens_gene_ensembl")
choice_list=c("entrezgene", "hgnc_symbol", "ensembl_gene_id", "unigene", "agilentprobe", 
              "affy_hg_u133_plus", "refseq_dna")
parser =  ArgumentParser()
parser$add_argument("original_id_type", type="character", nargs=1, metavar="ORIGINAL_GENE_ID_TYPE",
                    help="original ID to convert, supported gene types are ['entrezgene', 'hgnc_symbol', 'ensembl_gene_id', 'unigene', 'agilentprobe','affy_hg_u133_plus', 'refseq_dna']", 
                    choices=choice_list)
parser$add_argument("target_id_type", type="character", nargs=1, action="append", choices=choice_list,
                    metavar="TARGET_GENE_ID_TYPES",
                    help="types which you want to convert into, supported gene types are ['entrezgene', 'hgnc_symbol', 'ensembl_gene_id', 'unigene', 'agilentprobe','affy_hg_u133_plus', 'refseq_dna']")
parser$add_argument("--header", action="store_true", help="use this argument if your file has header")
parser$add_argument("--inplace", action="store_true", help="use this argument if you want to replace original file")
parser$add_argument("-c","--col", help="specify the number of column or header ID which has gene_id, default is 1",
                    default=1)
parser$add_argument("-o","--output", type="character", help="output file path")
parser$add_argument("-r","--replace", action="store_true", help="determine converted IDs will replace original IDs")
parser$add_argument("gene_ids", type="character", nargs="+", action="append",metavar="GENE_ID")
args = parser$parse_args()
#args = parser$parse_args(c("--header","--output", "C:/Users/dlsrnsi/Documents/csbl toolbox/test_2.txt" ,"--col","1","ensembl_gene_id", "hgnc_symbol", "C:/Users/dlsrnsi/Documents/csbl toolbox/test.txt"))
original_type =  args$original_id_type
target_type = args$target_id_type
file_name = args$gene_ids
header = args$header
col = args$col
replace = args$replace
if(suppressWarnings(!is.na(as.numeric(col)))){
  col =  as.numeric(col)
}
# read file
if(file_ext(file_name)=="csv"){
  gene_df = read.csv(file_name, header = header)
}else{
  gene_df = read.table(file_name,header = header)
}

gene_ids = gene_df[col]

# convert to target ID type
converted_ids = getBM(attributes=c(original_type,target_type), filters=original_type, values = gene_ids, mart=ensembl)
if(replace){
  gene_df[[col]] = converted_ids[match(gene_df[[col]], converted_ids[[original_type]]), target_type]
  result_df = gene_df
}else{
  result_df = merge(gene_df, converted_ids, by.x = col, by.y = original_type)
}

# make output
if(args$inplace){
  output = file_name
}else{
  output = args$output
}
if(file_ext(output)=="csv"){
  write.csv(result_df, file=output, row.names = FALSE)
}else{
  write.table(result_df, file=output, sep="\t", row.names = FALSE)
}
