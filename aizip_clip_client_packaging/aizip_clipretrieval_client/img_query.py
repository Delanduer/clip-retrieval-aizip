from .base_query import ClipBaseQuery

import base64
import os
from datetime import datetime
import requests

class ClipImgQuery(ClipBaseQuery):
    '''
    The class for image query
    '''
    def __init__(self) -> None:
        super().__init__()

    def query_for_image_dir(self, dir, num_res=1, indices_idx=0, print_res=False, save_in_file = None):
        '''
        Query interface for an entire directory of images

        :param dir: directory of images
        :type dir: str
        :param num_res: optional required number of results, in default 1
        :type num_res: int
        :param indices_idx: optional index number of the index to be used, in default 0
        :type indices_idx: int
        :param print_res: optinal switch for printing the results, in default False
        :type print_res: bool
        :param save_in_file: optional file path if results shall be saved
        :type save_in_file: str

        :returns: json encoded responses
        :rtype: list

        .. code-block:: python

           from aizip_clipretrieval_client import ClipImgQuery
           image_q_dir = img_query.query_for_image_dir(
                dir= "../testimages/",
                num_res= 1,
                indices_idx = 0,
                print_res=False,
                save_in_file = "./doc/test.html",
            )
        '''
        if len(self.indices) == 0:
            if self.debug_print:
                print("There is no valid index available on server. Return empty results.")
            return {}
        
        if os.path.exists(dir) == False:
           print("Given image folder doesn´t exist. Return empty result.")
           return {}
        # todo: extend other image formats
        from pathlib import Path
        images = Path(dir).glob("*.jpg")
        print("Accept only images with .jpg format!!")
        img_byte_list = []
        img_name_list = []
        for image in images:
            # encode image bytes with b42 first
            with open(image, "rb") as img_file:
                img_byte_list.append(self.__image_encoding(img_file.read()))
                img_name_list.append(image.name)

        return self.__query(
                images_string=str(img_byte_list),
                num_res = num_res,
                indices_idx = indices_idx,
                print_res = print_res,
                file_path = save_in_file,
            )

    def query_for_single_image(self, image_url, num_res=1, indices_idx=0, print_res=False, save_in_file=None):
        '''
        Query interface for single image. Not to be used directly.

        :param image_url: path of image to be queried
        :type image_url: str
        :param num_res: optional required number of results, in default 1
        :type num_res: int
        :param indices_idx: optional index number of the index to be used, in default 0
        :type indices_idx: int
        :param print_res: optinal switch for printing the results, in default False
        :type print_res: bool
        :param save_in_file: optional file path if results shall be saved
        :type save_in_file: str

        :returns: json encoded responses
        :rtype: list

        .. code-block:: python

           from aizip_clipretrieval_client import ClipImgQuery
           image_q = img_query.query_for_single_image(
                image_url = "./test.jpg",
                num_res= 1,
                print_res=False,
                save_in_file = "./doc/test.html",
            )
        '''        
        if os.path.exists(image_url) == False:
            print("Given image path is not valid, no image can be loaded. Return emtpy results.")
            return {}
        with open(image_url, "rb") as image:
            img_byte= [self.__image_encoding(image.read())]

        return self.__query(
            images_string=str(img_byte),
            num_res = num_res,
            indices_idx = indices_idx,
            print_res = print_res,
            file_path = save_in_file,
        )

    def __image_encoding(self, image, encoding="base64"):
        '''
        Encode the given image

        :param image: byte format of one image
        :type image: Bytes
        :param encoding: encoding type
        :type encoding: str

        :returns: encoded image bytes
        :rtype: Bytes
        '''
        if encoding == "base64":
            return base64.b64encode(image)
        else:
            print("This encoding: {} is not supported yet.".format(encoding))
            return None
        
    def __query(self, **kwargs):
        '''
        Override query function of image-query class

        :param `**kwargs`: The key word arguments of query attributes
        :returns: json encoded reponses of query
        :rtype: list
        '''
        assert kwargs["images_string"], f"'image_string' is not given as parameter for query in ClipImg class."
        
        total_url = self.host + ":" + self.port + "/knn-service"

        indice_name_str = self.get_index_name(kwargs["indices_idx"]) if kwargs["indices_idx"] else self.get_index_name(0)
        num_res_str = str(kwargs["num_res"]) if kwargs["num_res"] else "1"

        payload = {
            'query_images_list': (None, kwargs["images_string"]),
            'query_embeddings_list': (None, ''),
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
        self.post_print_and_save(response_json, file_path, print_out)    
        return response_json