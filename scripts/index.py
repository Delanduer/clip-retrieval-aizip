import argparse
from autofaiss import build_index
import time
import logging
import os

LOGGER = logging.getLogger(__name__)

def quantize(embeddings_folder, index_name, args):
    try:
        index_folder=args.out_path
        LOGGER.debug(
            f"starting building index {index_name}"
            f"using embeddings {embeddings_folder} ; saving in {index_folder}"
        )

        build_start = time.perf_counter()
        build_index(
            embeddings=embeddings_folder,
            index_path=index_folder + "/" + index_name + ".index",
            index_infos_path=index_folder + "/" + index_name + ".json",
            max_index_memory_usage=args.max_index_memory_usage if hasattr(args, 'max_index_memory_usage') else "32G",
            current_memory_available=args.current_memory_available if hasattr(args, 'current_memory_available') else "64G",
            nb_cores=args.nb_cores if hasattr(args, 'nb_cores') else None,
            max_index_query_time_ms=args.max_index_query_time_ms if hasattr(args, 'max_index_query_time_ms') else 10.0,
            min_nearest_neighbors_to_retrieve=args.min_nearest_neighbors_to_retrieve if hasattr(args, 'min_nearest_neighbors_to_retrieve') else 20,
            use_gpu=args.use_gpu if hasattr(args, 'use_gpu') else False,
            metric_type=args.metric_type if hasattr(args, 'metric_type') else "ip",
        )
        build_end = time.perf_counter()
        print("Total duration for building index of {}: {}".format(index_name, build_end-build_start))
        LOGGER.debug(f"index {index_name} done")
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.exception(f"index {index_name} failed")
        raise e

def callindex(args):
    if not os.path.exists(args.emb_path):
        assert "Given path for embedding is invalid"

    embs_folder_img = os.path.join(args.emb_path, "img_emb")
    embs_folder_txt = os.path.join(args.emb_path, "text_emb")

    if os.path.exists(embs_folder_img):
        quantize(embs_folder_img, "image", args)
    
    if os.path.exists(embs_folder_txt):
        quantize(embs_folder_txt, "text", args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_path", type=str, 
                        help="folder path of embeddings.")
    parser.add_argument("out_path", type=str,
                        help="path for index output.")
    parser.add_argument("--copy_metadata", type=bool, default=True,
                        help="whether or not to copy metadata")
    parser.add_argument("--max_index_memory_usage", type=str, default="16G",
                        help="maximum of memory to be used.")
    parser.add_argument("--current_memory_available", type=str, default="32G",
                        help="available memories.")
    parser.add_argument("--max_index_query_time_ms", type=float, default= 10.0,
                        help="max query time in ms as orientation.")
    parser.add_argument("--min_nearest_neighbors_to_retrieve", type=int, default= 20,
                        help="max query time in ms as orientation.")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="whether or not to use gpu.")
    
    args=parser.parse_args()
    print("Args: \n{}".format(args))
    build_start=time.perf_counter()
    callindex(args)
    build_end=time.perf_counter()
    print("Building index duration: {}".format(build_end-build_start))
