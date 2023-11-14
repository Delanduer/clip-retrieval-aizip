import requests

import pandas as pd
import os

class ClipBaseQuery:
    '''
    The abstract class for different query types. Not to be used directly.

    :ivar host: where the host url is stored
    :vartype host: str
    :ivar port: where the port number is stored
    :vartype port: str
    :ivar debug_print: whether to activate debug print
    :vartype debug_print: bool
    :ivar headers: common headers for http methods, including an integrated api token
    :vartype headers: dict
    :ivar indices: available indices on the remote server for clip retrieval
    :vartype indices: json
    '''
    def __init__(self):
        self.host = "http://12.12.12.13"
        self.port = "13005"
        self.debug_print = True

        #logging.basicConfig(
        #    filename='',
        #    encoding='utf-8',
        #    level=logging.DEBUG,
        #)
        self.headers = {
            'accept': 'application/json',
            'aizip-token': 'akljnv13bvi2vfo0b0bw',
            #'Content-Type': 'application/json',
        }
        #self.__get_request("/config.json")
        self.indices = self.__get_request("/indices-list")
    
    def __get_request(self, url, headers=None):
        '''
        Implmentation of sending HTTP GET request
        
        :param url: the actual resource locator
        :type url: str
        :param headers: optional header information
        :type headers: dict

        :returns: json encoded response of HTTP GET request
        :rtype: request.Response
        '''
        total_url = self.host + ":" + self.port + url
        response = requests.get(total_url, headers=self.headers)
        if self.debug_print:
            print("Available indices: {}".format(response))
        return response.json()
    
    def post_request(self, url, payload, headers=None):
        ''''''
        pass
    
    def get_index_name(self, indices_idx=0):
        '''
        Get full name of required index from indices list

        :param indices_idx: optional index number for the indices list
        :type indices_idx: int

        :returns: Full name of the required index
        :rtype: str
        '''
        indices_list = self.indices[indices_idx].strip().strip('["').strip('"]').split(",", -1) #.replace('\\\\','\\')
        index_name = indices_list[indices_idx] if indices_idx <= len(indices_list) else indices_list[0]
        if self.debug_print:
            print("index_name: {} with type: {}".format(index_name, type(index_name)))
        return index_name
    
    def __query(self, **kwargs):
        '''Virtual implementation of query function'''
        raise NotImplementedError
    
    def post_print_and_save(self, json_results, save_path=None, print_out=False):
        '''
        Wrapper for potential follow-up print of query results and save in HTML format

        :param json_results: json encoded query results
        :type json_results: request.Reponse
        :param save_path: optional path for saving results in HTML format
        :type save_path: str
        :param print_out: optional switch for printing out the results
        :type print_out: bool

        '''
        for idx, result in enumerate(json_results):
            if len(result) == 1:
                res_table = pd.DataFrame(
                    [(result[0]['image_path'], result[0]['id'], result[0]['similarity'])],
                    columns=["image_path", "id", "similarity"],
                )
                if print_out:
                    print("====================================")
                    print("======= results for image {} =======".format(idx))
                    print(res_table)
            else:
                res_table = pd.DataFrame(
                    [(e['image_path'], e['id'], e['similarity']) for e in result],
                    columns=["image_path", "id", "similarity"],
                )
                if print_out:
                    print("====================================")
                    print("======= results for image {} =======".format(idx))
                    print(res_table)
        
        if save_path is not None:
            os.makedirs(os.path.split(os.path.abspath(save_path))[0], exist_ok=True)
            with open(save_path, "w") as f:
                f.write(res_table.to_html())        