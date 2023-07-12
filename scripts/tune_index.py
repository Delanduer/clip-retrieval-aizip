from autofaiss import tune_index
import argparse


def tune_idx_py(args=argparse.Namespace):
    tune_index(
        index_path=args.index_path,
        index_key=args.index_key if hasattr(args, "index_key") else None,
        index_param=args.index_param,
        output_index_path=args.output_path,
        save_on_disk=True,
        min_nearest_neighbors_to_retrieve=args.min_nn if hasattr(args, "min_nn") else None,
        max_index_query_time_ms=args.max_query_ms if hasattr(args, "max_query_ms") else None,
        use_gpu=args.use_gpu if hasattr(args, "use_gpu") else False,
        verbose=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", type=str,
                        help="path of original index.")
    parser.add_argument("index_param", type=str,
                        help="string with hyperparameters to set to the index, separated by comma. Ex. 'nprobe=16,efSearch=32,ht=1024'")
    parser.add_argument("--index_key", type=str,
                        help="string to give to the index factory in order to create the index.")
    parser.add_argument("output_path", type=str,
                        help="path to store.")
    parser.add_argument("--min_nn", type=int,
                        help="Minimum number of nearest neighbors to retrieve when querying the index.")
    parser.add_argument("--max_query_ms", type=float,
                        help="Query speed constraint for the index to create.")
    parser.add_argument("--use_gpu", type=bool,
                        help="Experimental, gpu training is faster, not tested so far.")
    args = parser.parse_args()
    tune_idx_py(args)