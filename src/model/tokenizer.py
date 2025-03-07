"""
Tokenizer implementation for DistributedLLM.

Provides tokenization utilities for text processing in language models,
with support for various tokenization schemes.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Try to import the tokenizer libraries, with graceful fallbacks
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    logger.warning("SentencePiece not available. Some tokenizers may not work.")

try:
    import tokenizers
    from tokenizers import Tokenizer as HFTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logger.warning("Hugging Face tokenizers not available. Some tokenizers may not work.")


class Tokenizer:
    """
    Tokenizer for language models with support for various backends.
    
    This class provides a unified interface for different tokenization libraries,
    with support for batched operations and caching.
    """
    
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        legacy_mode: bool = False,
    ):
        """
        Initialize the tokenizer for a specific model.
        
        Args:
            model_id: ID of the model to load the tokenizer for
            cache_dir: Directory to cache tokenizer files
            legacy_mode: Whether to use legacy tokenization mode
        """
        self.model_id = model_id
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "distributed_llm")
        self.legacy_mode = legacy_mode
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default token IDs
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        
        # Load tokenizer based on model ID
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on model ID."""
        if "llama" in self.model_id.lower():
            self._load_llama_tokenizer()
        elif "t5" in self.model_id.lower():
            self._load_t5_tokenizer()
        elif "gpt" in self.model_id.lower():
            self._load_gpt_tokenizer()
        else:
            logger.warning(f"Unknown model type: {self.model_id}. Using fallback tokenizer.")
            self._load_fallback_tokenizer()
    
    def _load_llama_tokenizer(self):
        """Load LLaMA tokenizer."""
        if not TOKENIZERS_AVAILABLE:
            logger.error("LLaMA tokenizer requires 'tokenizers' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would download or locate the tokenizer files
            # For this mock implementation, we'll create a basic tokenizer
            self.tokenizer = self._create_mock_tokenizer()
            self.tokenizer_type = "llama"
            
            # Specific settings for LLaMA
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 0
            
            logger.info("Loaded LLaMA tokenizer")
        except Exception as e:
            logger.error(f"Error loading LLaMA tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_t5_tokenizer(self):
        """Load T5 tokenizer."""
        if not SENTENCEPIECE_AVAILABLE:
            logger.error("T5 tokenizer requires 'sentencepiece' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would load the SentencePiece model
            # For this mock implementation, we'll create a basic tokenizer
            self.sp_model = None  # This would be a SentencePiece model
            self.tokenizer_type = "t5"
            
            # Specific settings for T5
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.unk_token_id = 2
            
            logger.info("Loaded T5 tokenizer")
        except Exception as e:
            logger.error(f"Error loading T5 tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_gpt_tokenizer(self):
        """Load GPT tokenizer."""
        if not TOKENIZERS_AVAILABLE:
            logger.error("GPT tokenizer requires 'tokenizers' library.")
            self._load_fallback_tokenizer()
            return
        
        try:
            # In a real implementation, we would load the tokenizer files
            # For this mock implementation, we'll create a basic tokenizer
            self.tokenizer = self._create_mock_tokenizer()
            self.tokenizer_type = "gpt"
            
            # Specific settings for GPT
            self.bos_token_id = 50256
            self.eos_token_id = 50256
            self.pad_token_id = 50256
            
            logger.info("Loaded GPT tokenizer")
        except Exception as e:
            logger.error(f"Error loading GPT tokenizer: {e}")
            self._load_fallback_tokenizer()
    
    def _load_fallback_tokenizer(self):
        """Load a fallback tokenizer when the specific one is not available."""
        self.tokenizer_type = "fallback"
        logger.warning("Using fallback tokenizer with limited functionality")
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for demonstration purposes."""
        if TOKENIZERS_AVAILABLE:
            # Create a simple Hugging Face tokenizer
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import Whitespace
            
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            
            return tokenizer
        else:
            return None
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text string to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens like BOS/EOS
        
        Returns:
            List of token IDs
        """
        return self.encode_batch([text], add_special_tokens)[0]
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens like BOS/EOS
        
        Returns:
            List of token ID lists for each text
        """
        if self.tokenizer_type == "fallback":
            # Very basic fallback implementation
            return [[ord(c) % 30000 for c in text] for text in texts]
        
        elif self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if not TOKENIZERS_AVAILABLE:
                return self._fallback_encode_batch(texts, add_special_tokens)
            
            # Use Hugging Face tokenizers library
            encoding = self.tokenizer.encode_batch(texts)
            token_ids = [e.ids for e in encoding]
            
            # Add special tokens if requested
            if add_special_tokens:
                token_ids = [
                    [self.bos_token_id] + ids + [self.eos_token_id]
                    for ids in token_ids
                ]
            
            return token_ids
        
        elif self.tokenizer_type == "t5":
            if not SENTENCEPIECE_AVAILABLE or self.sp_model is None:
                return self._fallback_encode_batch(texts, add_special_tokens)
            
            # Use SentencePiece
            token_ids = []
            for text in texts:
                ids = self.sp_model.encode(text, out_type=int)
                if add_special_tokens:
                    ids = [self.bos_token_id] + ids + [self.eos_token_id]
                token_ids.append(ids)
            
            return token_ids
        
        else:
            return self._fallback_encode_batch(texts, add_special_tokens)
        
    def _fallback_encode_batch(self, texts: List[str], add_special_tokens: bool) -> List[List[int]]:
        """Fallback implementation for encoding when libraries aren't available."""
        token_ids = []
        for text in texts:
            # Simple character-based tokenization as fallback
            ids = [ord(c) % 30000 for c in text]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            token_ids.append(ids)
        
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens like BOS/EOS
        
        Returns:
            Decoded text string
        """
        return self.decode_batch([token_ids], skip_special_tokens)[0]

    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID lists back to texts.
        
        Args:
            batch_token_ids: List of token ID lists to decode
            skip_special_tokens: Whether to skip special tokens like BOS/EOS
        
        Returns:
            List of decoded text strings
        """
        if self.tokenizer_type == "fallback":
            # Very basic fallback implementation
            return ["".join(chr(min(id, 127)) for id in ids) for ids in batch_token_ids]
        
        elif self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if not TOKENIZERS_AVAILABLE:
                return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)
            
            # Filter special tokens if requested
            if skip_special_tokens:
                special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                filtered_ids = [
                    [id for id in ids if id not in special_tokens]
                    for ids in batch_token_ids
                ]
            else:
                filtered_ids = batch_token_ids
            
            # Use Hugging Face tokenizers
            texts = self.tokenizer.decode_batch(filtered_ids)
            return texts
        
        elif self.tokenizer_type == "t5":
            if not SENTENCEPIECE_AVAILABLE or self.sp_model is None:
                return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)
            
            # Filter special tokens if requested
            texts = []
            for ids in batch_token_ids:
                if skip_special_tokens:
                    special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                    ids = [id for id in ids if id not in special_tokens]
                
                text = self.sp_model.decode(ids)
                texts.append(text)
            
            return texts
        
        else:
            return self._fallback_decode_batch(batch_token_ids, skip_special_tokens)

    def _fallback_decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool) -> List[str]:
        """Fallback implementation for decoding when libraries aren't available."""
        texts = []
        for ids in batch_token_ids:
            # Filter special tokens if requested
            if skip_special_tokens:
                special_tokens = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
                ids = [id for id in ids if id not in special_tokens]
            
            # Simple character-based decoding as fallback
            text = "".join(chr(min(id, 127)) for id in ids)
            texts.append(text)
        
        return texts

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        if self.tokenizer_type == "llama" or self.tokenizer_type == "gpt":
            if hasattr(self.tokenizer, "get_vocab_size"):
                return self.tokenizer.get_vocab_size()
            return 32000  # Default approximate value
        
        elif self.tokenizer_type == "t5":
            if self.sp_model:
                return self.sp_model.vocab_size()
            return 32000  # Default approximate value
        
        return 30000  # Fallback approximate value