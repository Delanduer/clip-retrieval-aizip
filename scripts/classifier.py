
from embedding_reader import EmbeddingReader
import pandas as pd
import numpy as np
import argparse
import time

def main(
    emb_folder = None,
    file_format = "parquet_npy",
    embedding_column = "embedding",
    metadata_folder = None,
    meta_columns = ["image_path"],
    embedding_read_batch_size=10**6,
    classifier_weight_file = None,
    classifier_bias_file = None,
    print_debug_info = False,
    parquet_output_path = None,
):
    """
    main func to calculate the classification results.
    """
    start_t = time.perf_counter()
    embedding_reader = EmbeddingReader(
        embeddings_folder=emb_folder,
        file_format=file_format,
        embedding_column=embedding_column,
        metadata_folder=metadata_folder,
        meta_columns=meta_columns,
    )
    end_read = time.perf_counter()
    print("Duration for reading embeddings: {}".format(end_read - start_t))
    
    for embs, meta in embedding_reader(
        batch_size=embedding_read_batch_size,
        start=0,
        end=embedding_reader.count
    ):
        pass
        #print("Shape of read embedding: {}".format(embs.shape))

    l14_weight = pd.read_csv(classifier_weight_file, sep=" ", header=None)
    l14_bias = pd.read_csv(classifier_bias_file, sep=" ", header=None)

    if print_debug_info:
        print("Shape of weight: {}".format(l14_weight.shape))
        print("Shape of bias: {}".format(l14_bias.shape))
    
    total_results = []
    calc_start = time.perf_counter()
    for idx, emb in enumerate(embs):
        single_dict = {}
        res_tmp = np.add(np.dot(l14_weight, emb).reshape(-1,1), l14_bias.to_numpy())
        single_dict["classifier_tmp_res"] = res_tmp.tolist()
        single_dict["classifier_res"] = 0 if res_tmp[0][0] > res_tmp[1][0] else 1
        single_dict["image_path"] = meta["image_path"][idx]
        total_results.append(single_dict)
    calc_end = time.perf_counter()
    print("Duration for classifier calculation: {}".format(calc_end - calc_start))

    if print_debug_info:
         print("Total length of results: {}".format(len(total_results)))
         print("Sample data: \n{}".format(total_results[0]))
    
    write_start = time.perf_counter()
    write_df = pd.DataFrame.from_dict(total_results)
    write_df.to_parquet(parquet_output_path)
    write_end = time.perf_counter()
    print("Duration for writing parquet file: {}".format(write_end - write_start))


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   import pathlib
   parser.add_argument("emb_folder", type= str,
                       help="Abs. path of embedding folder.")
   parser.add_argument("meta_folder", type=str,
                       help="Abs. path of metadata folder.")
   parser.add_argument("output_path", type=str,
                       help="Abs. path for output parquet file.")
   parser.add_argument("weight", type=str,
                       help="Abs. path of classifier weight file.")
   parser.add_argument("bias", type=str,
                       help="Abs. path of classifier bias file.")
   args=parser.parse_args()
   main(
       emb_folder=args.emb_folder,
       metadata_folder=args.meta_folder,
       parquet_output_path=args.output_path,
       classifier_weight_file=args.weight,
       classifier_bias_file=args.bias,
   )
