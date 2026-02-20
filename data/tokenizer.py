"""
Production-grade BPE Tokenizer implementation.

Features:
- Trainable BPE (Byte Pair Encoding)
- Efficient vocabulary management
- Special token handling
- Save/load functionality
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    max_length: int = 512
    
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"
    
    lowercase: bool = True
    normalization: str = "nfc"
    
    min_frequency: int = 2
    
    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.mask_token]


class Tokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Set[str] = set()
        
        self._init_special_tokens()
    
    def _init_special_tokens(self) -> None:
        for i, token in enumerate(self.config.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.config.pad_token]
    
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.config.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.config.eos_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.config.unk_token]
    
    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.config.mask_token]
    
    def normalize(self, text: str) -> str:
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.normalization:
            try:
                text = unicodedata.normalize(self.config.normalization, text)
            except (ValueError, TypeError):
                pass
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        raise NotImplementedError
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        raise NotImplementedError
    
    def save(self, path: Path) -> None:
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer with full training support.
    """
    
    WORD_SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""", re.UNICODE)
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )
    
    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _tokenize_word(self, word: str) -> List[str]:
        if word in self.token_to_id:
            return [word]
        
        word_tokens = tuple(self.byte_encoder[b] for b in word.encode('utf-8'))
        
        if len(word_tokens) == 0:
            return []
        
        while len(word_tokens) > 1:
            pairs = self._get_pairs(word_tokens)
            
            if not pairs:
                break
            
            bigram = min(pairs, key=lambda p: self.merges.index(p) if p in self.merges else float('inf'))
            
            if bigram not in self.merges:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word_tokens):
                try:
                    j = word_tokens.index(first, i)
                except ValueError:
                    new_word.extend(word_tokens[i:])
                    break
                
                new_word.extend(word_tokens[i:j])
                i = j
                
                if i < len(word_tokens) - 1 and word_tokens[i] == first and word_tokens[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            word_tokens = tuple(new_word)
        
        return list(word_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        tokens = []
        
        for match in re.finditer(self.pattern, text):
            word = match.group(0)
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        tokens = self.tokenize(text)
        
        token_ids = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)
        
        if add_eos:
            token_ids.append(self.eos_token_id)
        
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length - (1 if add_eos else 0)]
                if add_eos:
                    token_ids[-1] = self.eos_token_id
            elif padding:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.config.special_tokens:
                    continue
                tokens.append(token)
        
        text = ''.join(tokens)
        decoded = bytearray([self.byte_decoder[c] for c in text if c in self.byte_decoder]).decode('utf-8', errors='replace')
        
        return decoded
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> None:
        logger.info(f"Training BPE tokenizer on {len(texts)} texts...")
        
        word_freqs: Counter = Counter()
        for text in texts:
            text = self.normalize(text)
            for match in re.finditer(self.pattern, text):
                word = match.group(0)
                word_bytes = tuple(self.byte_encoder[b] for b in word.encode('utf-8'))
                word_freqs[word_bytes] += 1
        
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        vocab.update(self.config.special_tokens)
        
        merges: List[Tuple[str, str]] = []
        num_merges = self.config.vocab_size - len(self.config.special_tokens) - len(vocab)
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(num_merges), desc="Training BPE")
        else:
            iterator = range(num_merges)
        
        for _ in iterator:
            pair_freqs: Counter = Counter()
            
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merges.append(best_pair)
            
            new_word_freqs: Counter = Counter()
            bigram = best_pair
            replacement = bigram[0] + bigram[1]
            
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                        new_word.append(replacement)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] += freq
            
            word_freqs = new_word_freqs
            vocab.add(replacement)
        
        self.merges = merges
        self.vocab = vocab
        
        for i, token in enumerate(sorted(vocab)):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        logger.info(f"Trained tokenizer with {self.vocab_size} tokens, {len(self.merges)} merges")
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": {
                "vocab_size": self.config.vocab_size,
                "max_length": self.config.max_length,
                "pad_token": self.config.pad_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "unk_token": self.config.unk_token,
                "mask_token": self.config.mask_token,
                "lowercase": self.config.lowercase,
                "normalization": self.config.normalization,
                "min_frequency": self.config.min_frequency,
            },
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "vocab": list(self.vocab),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved tokenizer to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        path = Path(path)
        
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        config = TokenizerConfig(**data["config"])
        tokenizer = cls(config)
        
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab = set(data["vocab"])
        
        return tokenizer
    
    @classmethod
    def from_pretrained(cls, name: str, cache_dir: Optional[Path] = None) -> "BPETokenizer":
        cache_dir = cache_dir or Path.home() / ".cache" / "multimodal_tokenizers"
        tokenizer_path = cache_dir / name / "tokenizer.json"
        
        if tokenizer_path.exists():
            logger.info(f"Loading cached tokenizer from {tokenizer_path}")
            return cls.load(tokenizer_path)
        
        raise FileNotFoundError(
            f"Tokenizer '{name}' not found in cache. "
            f"Train a new tokenizer or download it to {tokenizer_path}"
        )
