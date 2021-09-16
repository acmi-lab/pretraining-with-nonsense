import logging
import numpy as np
import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class SubsequenceDatasetReader(DatasetReader):
    '''reads a subsequence of datapoints from the input file defined by a seed and max_to_read parameters.
    For test sets, you can pass -1 to both of them to override this behavior'''
    def __init__(self,
                 lazy:bool = False,
                 random_seed:int = -1,
                 max_to_read:int = -1) -> None:
        if random_seed!=-1:
            lazy=False  # have to read the full file to get all datapoints before getting subsequence
        super().__init__(lazy=lazy)
        self.random_seed = random_seed
        self.max_to_read = max_to_read

    @overrides
    def _read(self, file_path:str):
        logger.info("Reading instances from lines in file at: %s", file_path)
        saved_objs = []
        num_passed=0

        limit=0
        if self.max_to_read==-1:
            limit = np.inf
        elif self.max_to_read>=0:
            limit = self.max_to_read
        else:
            raise ArithmeticError(f"Invalid value of max_to_read parameter : {self.max_to_read}")

        with open(file_path, "r") as r:
            for line in r:
                if num_passed == limit:
                    break
                l=line.strip()
                if len(l)==0:
                    continue
                else:
                    dp = json.loads(l)
                    num_passed += 1
                    saved_objs.append(dp)

        if self.random_seed!=-1:
            gen = np.random.default_rng(self.random_seed)
            perm = gen.permutation(len(saved_objs))
        else:
            perm = range(len(saved_objs))

        for index in perm:
            instance = self.dict_to_instance(saved_objs[index])
            if instance!=None:
                yield instance

    def dict_to_instance(self, json_dict:dict):
        '''converts the read json object to an allennlp Instance object'''
        raise NotImplementedError



