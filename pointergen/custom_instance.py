from allennlp.data import Instance
from overrides import overrides
from allennlp.data import Vocabulary
from pointergen.fields import SourceTextField, TargetTextField

class SyncedFieldsInstance(Instance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @overrides
    def index_fields(self, vocab: Vocabulary) -> None:
        if not self.indexed:
            self.indexed = True
            all_fields = self.fields.values()
            source_fields = list(filter(lambda x:type(x)==SourceTextField, all_fields))
            target_fields = list(filter(lambda x:type(x)==TargetTextField, all_fields))

            assert (len(source_fields)==1), "There should be exactly one source fields because otherwise OOV indices would clash"
            for field in self.fields.values():
                if type(field) not in [SourceTextField, TargetTextField]:
                    field.index(vocab)

            source_field = source_fields[0]
            oov_list = source_field.index(vocab)
            self.oov_list = oov_list

            for target_field in target_fields:
                target_field.index(vocab, oov_list)



