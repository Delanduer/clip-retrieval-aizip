from .base_query import ClipBaseQuery

import os
import numpy as np
from datetime import datetime
import requests

class ClipClassifierQuery(ClipBaseQuery):
    '''
    The class for classifier query
    '''
    def __init__(self) -> None:
        super().__init__()

    def query_for_csv(self, csv_file, csv_idx=[0], num_res=1, indices_idx=0, print_res=False, save_in_file=None):
        '''
        Query interface for embeddings stored in csv file

        :param csv_file: path of csv file
        :type csv_file: str
        :param csv_idx: index of embeddings to be used from csv file
        :type csv_idx: list
        :param num_res: optional required number of results, in default 1
        :type num_res: int
        :param indices_idx: optional index number of the index to be used, in default 0
        :type indices_idx: int
        :param print_res: optinal switch for printing the results, in default False
        :type print_res: bool
        :param save_in_file: optional file path if results shall be saved
        :type save_in_file: str

        :returns: json encoded responses of request
        :rtype: list

        .. code-block:: python

           from aizip_clipretrieval_client import ClipClassifierQuery
           csv_embeddings = classifier_query.query_for_csv(
                csv_file = "./xx.csv",
                csv_idx=[0,1],
                num_res=1, 
                indices_idx=0, 
                print_res=False, 
                save_in_file="./doc/test.html",
            )
        '''        
        if os.path.exists(csv_file) == False:
           print("Given csv file doesn´t exist. Return empty result.")
           return {}
        csv_emb = self.__load_emb_from_csv(csv_file, ' ')
        if csv_emb is None:
            print("No valid embeddings can be read from given .csv file.")
            return []
        embeddings = []
        for idx in csv_idx:
            if idx < csv_emb.shape[0]:
                embeddings.append(self.__convert_emb_to_str(csv_emb[idx,:]))
        return self.__query(
            embeddings_str=str(embeddings),
            num_res=num_res,
            indices_idx=indices_idx,
            print_res=print_res,
            file_path = save_in_file,
        )        

    def query_for_single_embedding(self, embedding, num_res=1, indices_idx=0, print_res=False, save_in_file=None):
        '''
        Query interface for single embedding of classifier

        :param embedding: embedding of classifier   
        :type embedding: bytes
        :param num_res: optional required number of results, in default 1
        :type num_res: int
        :param indices_idx: optional index number of the index to be used, in default 0
        :type indices_idx: int
        :param print_res: optinal switch for printing the results, in default False
        :type print_res: bool
        :param save_in_file: optional file path if results shall be saved
        :type save_in_file: str

        :returns: json encoded responses of query
        :rtype: list

        .. code-block:: python

           from aizip_clipretrieval_client import ClipClassifierQuery
           q_emb = classifier_query.query_for_single_embedding(
                embedding = [-1.66662186e-02, -1.46154482e-02, -6.44422323e-02, -3.64316478e-02,
                ...
                -6.18078327e-03, -4.24819812e-02, -3.35807130e-02, -3.08844335e-02],
                num_res=1, 
                indices_idx=0, 
                print_res=False, 
                save_in_file="./doc/test.html",
            )
        '''
        return self.__query(
            embeddings_str = str([self.__convert_emb_to_str(embedding)]),
            num_res = num_res, 
            indices_idx = indices_idx,
            print_res=print_res,
            file_path = save_in_file,
        )
    
    def query_for_embeddings(self, embeddings, num_res=1, indices_idx=0, print_res=False, save_in_file=None):
        '''
        Query interface for multiple classifier embeddings

        :param embeddings: list of classifier embeddings   
        :type embeddings: list
        :param num_res: optional required number of results, in default 1
        :type num_res: int
        :param indices_idx: optional index number of the index to be used, in default 0
        :type indices_idx: int
        :param print_res: optinal switch for printing the results, in default False
        :type print_res: bool
        :param save_in_file: optional file path if results shall be saved
        :type save_in_file: str

        :returns: json encoded responses of query
        :rtype: list

        .. code-block:: python

           from aizip_clipretrieval_client import ClipClassifierQuery
           q_embs = classifier_query.query_for_embeddings(
                embeddings = [
                    [-1.66662186e-02, ..., -3.64316478e-02,],
                    [-6.18078327e-03, ..., -3.08844335e-02]
                ],
                num_res=1, 
                indices_idx=0, 
                print_res=False, 
                save_in_file="./doc/test.html",
            )
        '''
        embeddings_list = [self.__convert_emb_to_str(embedding) for embedding in embeddings]
        return self.__query(
            embeddings_str = str(embeddings_list),
            num_res = num_res, 
            indices_idx = indices_idx,
            print_res=print_res,
            file_path = save_in_file,
        )

    def __load_emb_from_csv(self, csv_file, sep=' '):
        '''
        Load embeddings from given csv file 

        :param csv_file: path of csv file
        :type csv_file: str
        :param sep: delimeter in csv file
        :type sep: str

        :returns: embeddings of csv file
        :rtype: ndarray
        '''
        return np.genfromtxt(csv_file, delimiter=sep)

    def __convert_emb_to_str(self, embedding):
        '''
        Convert embedding bytes to string format

        :param embedding: whole embedding
        :type embedding: ndarray

        :returns: embedding in string format
        :rtype: list
        '''
        return ["%.10f" % number for number in embedding]

    def __query(self, **kwargs):
        '''
        Override query function of classifier class

        :param `**kwargs`: The key word arguments of query attributes
        :returns: json encoded reponses of query
        :rtype: list
        '''
        assert kwargs["embeddings_str"], f"'embeddings_str' is not given as parameter for query in ClipImg class."
        total_url = self.host + ":" + self.port + "/knn-service"
        indice_name_str = self.get_index_name(kwargs["indices_idx"]) if kwargs["indices_idx"] else self.get_index_name(0)
        num_res_str = str(kwargs["num_res"]) if kwargs["num_res"] else "1"

        payload = {
            'query_embeddings_list': (None, kwargs["embeddings_str"]),
            'query_images_list': (None, ''),
            'modality': (None, 'image'),
            'use_mclip': (None, 'false'),
            'deduplicate': (None, 'false'),
            'num_images': (None, num_res_str),
            'use_violence_detector': (None, 'false'),
            'use_safety_model': (None, 'false'),
            'indice_name': (None, indice_name_str),
        }
        
        query_start_time = datetime.now()
        response = requests.post(total_url, files=payload, headers=self.headers)
        query_end_time = datetime.now()
        response_json = response.json()
        if self.debug_print:
            print("Status code: {}".format(response.status_code))
        print("Total time for query: {}".format(query_end_time - query_start_time))

        file_path = kwargs["file_path"] if kwargs["file_path"] else None
        print_out = kwargs["print_res"] if kwargs["print_res"] else False
        self.post_print_and_save(
            json_results=response_json,
            save_path=file_path,
            print_out=print_out,
        )

        return response_json