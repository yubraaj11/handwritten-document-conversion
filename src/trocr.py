# IIT =- TrOCR
from transformers import VisionEncoderDecoderModel
from transformers import ViTImageProcessor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image

class HandwrittenTextProcessor:

    def __init__(self,encode_model: str, decode_model: str):
        """ Initialize the processser with feature extractor and tokenizer"""
        self.tokenizer = RobertaTokenizer.from_pretrained(decode_model)
        self.feature_extractor = ViTImageProcessor.from_pretrained(encode_model)
        self.processor = TrOCRProcessor(image_processor=self.feature_extractor, tokenizer=self.tokenizer) 

    def get_processor(self):
        """ Returns the processor with feature extractor and tokenizer"""
        return self.processor
    
class HandwrittenTextModel:

    def __init__(self,encode_model: str,decode_model:str=None, pre_trained_model : str = None):
        """ Initialize the model with encoder and decoder"""
        if pre_trained_model:
            self.model = VisionEncoderDecoderModel.from_pretrained(pre_trained_model)
        else:
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode_model, decode_model)
    
    
    def configure_model(self, processor:TrOCRProcessor, config:dict):
        """ Configure the model with processor and config"""
        self.model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Set beam search parameters
        self.model.config.eos_token_id = processor.tokenizer.sep_token_id
        self.model.config.max_length = config.get('max_length', 16)
        self.model.config.early_stopping = config.get('early_stopping', True)
        self.model.config.no_repeat_ngram_size = config.get('no_repeat_ngram_size', 3)
        self.model.config.length_penalty = config.get('length_penalty', 2.0)
        self.model.config.num_beams = config.get('num_beams', 4)

    def get_model(self):
        return self.model
    
