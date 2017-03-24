import pandas as pd


def gramize(seq, n):
    """
    make n-gram with seq with offset [0,...,n-1]

    Args:
        seq: protein sequence
        n : the number of gram
    Return:
        grams: list of list of n-gram by offset
    """
    if pd.isnull(seq):
        return None
    grams = []
    n_gram = len(seq)-n+1
    for offset in range(n):
        grams_offset = []
        for char in range(n_gram):
            if char%n==offset:
                grams_offset.append(seq[char:char+n])
        grams.append(grams_offset)
    return grams


def flatten_list(list_of_list):
    """
    flatten list e.g [['a',b'],['c']] -> ['a','b','c']

    Args:
        list_of_list list: list include list
    Return:
        flattend list: flattened list
    """
    flattened_list = []
    for list_element in list_of_list:
        if list_element:
            flattened_list += list_element
        else:
            continue

    return flattened_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="the number of gram", default=333, type=int)
    parser.add_argument("input_file", help="input file to gramize")
    parser.add_argument("output_file", help="output file")
    parser.add_argument("--index", help="use if you have index")
    parser.add_argument("--header", action="store_true", help="if file has header, use this argument")
    parser.add_argument("--gram-only", help="save gram only", action="store_true")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    index = args.index
    gram_only = args.gram_only
    n = int(args.n)

    if index:
        df = None
        if args.header:
            if input_file.endswith(".csv"):
                df = pd.read_csv(input_file)
            else:
                df = pd.read_table(input_file)
        else:
           if input_file.endswith(".csv"):
                df = pd.read_csv(input_file, header=None)
           else:
                df = pd.read_table(input_file, header=None)
        df[str(n)+"_gram"] = df[index].map(lambda seq: gramize(seq, n))
        df.dropna(inplace=True)
        if gram_only:
            gram_list = df[str(n)+"_gram"].tolist()
            gram = flatten_list(gram_list)
            f = open(output_file, 'w')
            f.writelines(list(map(lambda n_grams: " ".join(n_grams),gram)))
            f.close()
        else:
            if output_file.endswith(".csv"):
                df.to_csv(output_file, index=False, index_label=None)
            else:
                df.to_csv(output_file, sep="\t", index=False, index_label=None)
    else:
        f = open(input_file)
        seq_list = f.readlines()
        f.close()
        gram = map(lambda seq: gramize(seq, n), seq_list)
        gram = flatten_list(gram)
        f = open(output_file, 'w')
        f.writelines(gram)
        f.close()
