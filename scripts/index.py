import argparse
from clip_retrieval.clip_index import clip_index
import time

def callindex():
    clip_index(
        embeddings_folder=args.emb_path,
        index_folder=args.out_path,
        copy_metadata=args.copy_metadata if args.copy_metadata else True ,
        max_index_memory_usage=args.max_index_memory_usage if args.max_index_memory_usage else "16G",
        current_memory_available=args.current_memory_available if args.current_memory_available else "64G",
    )



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_path", type=str, 
                        help="folder path of embeddings.")
    parser.add_argument("out_path", type=str,
                        help="path for index output.")
    parser.add_argument("--copy_metadata", type=bool,
                        help="whether or not to copy metadata")
    parser.add_argument("--max_index_memory_usage", type=str,
                        help="maximum of memory to be used.")
    parser.add_argument("--current_memory_available", type=str,
                        help="available memories.")
    
    args=parser.parse_args()
    build_start=time.perf_counter()
    callindex(args)
    build_end=time.perf_counter()
    print("Building index duration: {}".format(build_end-build_start))
