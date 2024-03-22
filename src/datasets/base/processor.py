from src.tokenizers.models.info import Info
from src.common.models.verification import Verification
from src.common.models.args_info import ArgsInfo
from src.common.services.verification import verify_args

class Processor:

    def __init__(self, info: Info):
        self.info = info
        self.tokenizer = info["tokenizer"]
        self.pad_idx = info["pad_idx"]
        self.bos_idx = info["bos_idx"]
        self.eos_idx = info["eos_idx"]
        self.mask_idx = info["mask_idx"]
        self.vocab_size = info["vocab_size"]

    def process(self, **kwargs):
        raise NotImplementedError
    
    def validate_args(self, **kwargs) -> Verification:
        return verify_args({}, **kwargs)

    def encode(self, sample):
        raise NotImplementedError