import logging
import pdb
from typing import List, Dict

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from overrides import overrides

from pointergen.custom_instance import SyncedFieldsInstance
from pointergen.fields import SourceTextField, TargetTextField
from utils.subsequence_dataset_reader import SubsequenceDatasetReader
from transformers import T5Tokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("pretrained_wordpiece_dataset_reader")
class PretrainedCNNDmailDatasetReader(SubsequenceDatasetReader):
    def __init__(self,
                 max_source_length : int = None,
                 max_target_length : int =None,
                 max_source_wpiece_length: int=None,
                 max_target_wpiece_length: int=None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lowercase_tokens : bool = False,
                 lazy: bool = False,
                 random_seed: int=-1,
                 max_to_read:int = -1) -> None:
        super().__init__(lazy=lazy, random_seed=random_seed, max_to_read=max_to_read)
        self.lowercase_tokens = lowercase_tokens
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_source_wpiece_length = max_source_wpiece_length
        self.max_target_wpiece_length = max_target_wpiece_length

        # REMEMBER : Your data file must not contain things like PAD, UNK, START, STOP explicitly
        self._tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._token_indexers or \
                not isinstance(self._token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CNNDmailDatasetReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")


    @overrides
    def dict_to_instance(self, dp):
        source_sequence = " ".join(dp["article_lines"])
        target_sequence = " ".join(dp["summary_lines"])
        if len(source_sequence)>0 and len(target_sequence)>0:
            if self.max_source_length!=None:
                source_sequence = source_sequence.split(" ")[:self.max_source_length]
                source_sequence = " ".join(source_sequence)
            if self.max_target_length!=None:
                target_sequence = target_sequence.split(" ")[:self.max_target_length]
                target_sequence = " ".join(target_sequence)
            return self.text_to_instance(source_sequence, target_sequence)


    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        # pylint: disable=arguments-differ
        if self.lowercase_tokens:
            source_string = source_string.lower()
            target_string = target_string.lower()
        tokenized_source = self._tokenizer.tokenize(source_string)
        if self.max_source_wpiece_length!=None:
            tokenized_source = tokenized_source[:self.max_source_wpiece_length]
        tokenized_source = [Token(w) for w in tokenized_source]
        source_field = SourceTextField(tokenized_source, self._token_indexers)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source]}
        fields_dict = {
                "source_tokens": source_field,
        }

        if target_string is not None:
            tokenized_target = self._tokenizer.tokenize(target_string)
            tokenized_target = [Token(w) for w in tokenized_target]
            meta_fields["target_tokens"] = [x.text for x in tokenized_target]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            if self.max_target_wpiece_length!=None:
                tokenized_target = tokenized_target[:self.max_target_wpiece_length]
            target_field = TargetTextField(tokenized_target, self._token_indexers)
            fields_dict["target_tokens"] = target_field

        fields_dict["meta"] = MetadataField(meta_fields)

        return SyncedFieldsInstance(fields_dict)

