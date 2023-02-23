from embedding_reader import EmbeddingReader
import pandas as pd
import numpy as np

def main(
    embeddings_folder = None,
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
    embedding_reader=EmbeddingReader(
        embeddings_folder = embeddings_folder,
        file_format = file_format,
        embedding_column = embedding_column,
        metadata_folder = metadata_folder,
        meta_columns = meta_columns,
    )
    nb_vectors = embedding_reader.count
    vec_dim = embedding_reader.dimension
    
    for embs, meta in embedding_reader(
        batch_size=embedding_read_batch_size,
        start=0,
        end=embedding_reader.count
    ):
        print("Shape of read embedding: {}".format(embs.shape))

    l14_weight = pd.read_csv(classifier_weight_file, sep=" ", header=None)
    l14_bias = pd.read_csv(classifier_bias_file, sep=" ", header=None)

    if print_debug_info:
        print("Shape of weight: {}".format(l14_weight.shape))
        print("Shape of bias: {}".format(l14_bias.shape))
    
    total_results = []
    for idx, emb in enumerate(embs):
        single_dict = {}
        res_tmp = np.add(np.dot(l14_weight, emb).reshape(-1,1), l14_bias.to_numpy())
        single_dict["classifier_tmp_res"] = res_tmp.tolist()
        single_dict["classifier_res"] = 0 if res_tmp[0][0] > res_tmp[1][0] else 1
        single_dict["image_path"] = meta["image_path"][idx]
        total_results.append(single_dict)

    if print_debug_info:
         print("Total length of results: {}".format(len(total_results)))
         print("Sample data: \n{}".format(total_results[0]))

    write_df = pd.DataFrame.from_dict(total_results)
    write_df.to_parquet(parquet_output_path)


if __name__ == "__main__":
   main(
       embeddings_folder = "/home/junjie/test/multitaskgpupara/emb_laioni24_l14_b1024_700/img_emb",
       metadata_folder = "/home/junjie/test/multitaskgpupara/emb_laioni24_l14_b1024_700/metadata",
       parquet_output_path = "/home/junjie/test/multitaskgpupara/emb_laioni24_l14_b1024_700/classifier_res.parquet",
       classifier_weight_file = "/home/junjie/git/clip-retrieval-aizip/classifier/l14_weight.csv",
       classifier_bias_file = "/home/junjie/git/clip-retrieval-aizip/classifier/l14_bias.csv",
   )
