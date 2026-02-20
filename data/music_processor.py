"""
Music tokenization and processing for symbolic music generation.

Implements a MIDI-like token vocabulary:
- Note-on events (pitch + velocity)
- Note-off events
- Time-shift events
- Special tokens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class MusicProcessorConfig:
    vocab_size: int = 512
    
    midi_min_pitch: int = 21
    midi_max_pitch: int = 108
    
    velocity_bins: int = 32
    time_step_ms: int = 10
    max_time_steps: int = 1000
    
    max_seq_len: int = 1024
    
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3
    
    @property
    def pitch_range(self) -> int:
        return self.midi_max_pitch - self.midi_min_pitch + 1
    
    @property
    def num_note_on_tokens(self) -> int:
        return self.pitch_range * self.velocity_bins
    
    @property
    def num_note_off_tokens(self) -> int:
        return self.pitch_range
    
    @property
    def num_time_shift_tokens(self) -> int:
        return self.max_time_steps


class MusicProcessor:
    """
    Production-grade music processor for symbolic music tokenization.
    
    Token vocabulary:
    - 0: PAD
    - 1: BOS
    - 2: EOS
    - 3-2+pitch_range*velocity_bins: NOTE_ON (pitch * velocity_bins + velocity)
    - NOTE_ON_end+1 to NOTE_ON_end+pitch_range: NOTE_OFF
    - Remaining: TIME_SHIFT
    """
    
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    
    def __init__(self, config: Optional[MusicProcessorConfig] = None):
        self.config = config or MusicProcessorConfig()
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        cfg = self.config
        
        self._note_on_offset = 3
        self._note_off_offset = self._note_on_offset + cfg.num_note_on_tokens
        self._time_shift_offset = self._note_off_offset + cfg.num_note_off_tokens
        
        self._vocab_size = self._time_shift_offset + cfg.max_time_steps
        
        self._pitch_to_note_on = {}
        self._note_on_to_pitch = {}
        self._pitch_to_note_off = {}
        self._note_off_to_pitch = {}
        
        for pitch in range(cfg.midi_min_pitch, cfg.midi_max_pitch + 1):
            pitch_idx = pitch - cfg.midi_min_pitch
            for vel in range(cfg.velocity_bins):
                token = self._note_on_offset + pitch_idx * cfg.velocity_bins + vel
                self._pitch_to_note_on[(pitch, vel)] = token
            
            note_off_token = self._note_off_offset + pitch_idx
            self._pitch_to_note_off[pitch] = note_off_token
    
    @property
    def vocab_size(self) -> int:
        return min(self._vocab_size, self.config.vocab_size)
    
    def _quantize_velocity(self, velocity: int) -> int:
        return min(velocity * self.config.velocity_bins // 128, self.config.velocity_bins - 1)
    
    def _quantize_time(self, time_ms: float) -> int:
        step = int(time_ms / self.config.time_step_ms)
        return min(step, self.config.max_time_steps - 1)
    
    def encode_note_on(self, pitch: int, velocity: int) -> int:
        vel_bin = self._quantize_velocity(velocity)
        return self._pitch_to_note_on.get((pitch, vel_bin), self.config.unk_token_id)
    
    def encode_note_off(self, pitch: int) -> int:
        return self._pitch_to_note_off.get(pitch, self.config.unk_token_id)
    
    def encode_time_shift(self, time_ms: float) -> int:
        step = self._quantize_time(time_ms)
        return self._time_shift_offset + step
    
    def decode_token(self, token: int) -> Dict[str, Any]:
        if token == self.PAD_TOKEN:
            return {"type": "pad"}
        elif token == self.BOS_TOKEN:
            return {"type": "bos"}
        elif token == self.EOS_TOKEN:
            return {"type": "eos"}
        elif token < self._note_off_offset:
            rel_token = token - self._note_on_offset
            pitch_idx = rel_token // self.config.velocity_bins
            vel_bin = rel_token % self.config.velocity_bins
            pitch = self.config.midi_min_pitch + pitch_idx
            velocity = int((vel_bin + 0.5) * 128 / self.config.velocity_bins)
            return {"type": "note_on", "pitch": pitch, "velocity": velocity}
        elif token < self._time_shift_offset:
            pitch_idx = token - self._note_off_offset
            pitch = self.config.midi_min_pitch + pitch_idx
            return {"type": "note_off", "pitch": pitch}
        else:
            step = token - self._time_shift_offset
            time_ms = step * self.config.time_step_ms
            return {"type": "time_shift", "time_ms": time_ms}
    
    def encode_midi_events(
        self,
        events: List[Dict[str, Any]],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = []
        
        if add_bos:
            tokens.append(self.BOS_TOKEN)
        
        current_time = 0
        for event in events:
            event_type = event.get("type")
            
            if event_type == "note_on":
                time_diff = event.get("time", current_time) - current_time
                if time_diff > 0:
                    tokens.append(self.encode_time_shift(time_diff * 1000))
                    current_time = event.get("time", current_time)
                
                pitch = event.get("pitch", 60)
                velocity = event.get("velocity", 80)
                tokens.append(self.encode_note_on(pitch, velocity))
            
            elif event_type == "note_off":
                time_diff = event.get("time", current_time) - current_time
                if time_diff > 0:
                    tokens.append(self.encode_time_shift(time_diff * 1000))
                    current_time = event.get("time", current_time)
                
                pitch = event.get("pitch", 60)
                tokens.append(self.encode_note_off(pitch))
        
        if add_eos:
            tokens.append(self.EOS_TOKEN)
        
        max_len = max_length or self.config.max_seq_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [self.EOS_TOKEN]
        
        return tokens
    
    def decode_tokens(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True,
    ) -> List[Dict[str, Any]]:
        events = []
        current_time = 0
        
        for token in tokens:
            decoded = self.decode_token(token)
            event_type = decoded.get("type")
            
            if event_type in ("pad", "bos") and skip_special_tokens:
                continue
            elif event_type == "eos":
                break
            elif event_type == "time_shift":
                current_time += decoded.get("time_ms", 0) / 1000.0
            elif event_type in ("note_on", "note_off"):
                decoded["time"] = current_time
                events.append(decoded)
        
        return events
    
    def process(
        self,
        source: Union[List[Dict], List[int], Tensor],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> Tensor:
        if isinstance(source, Tensor):
            if source.dim() == 0:
                source = source.unsqueeze(0)
            return source
        
        if isinstance(source, list):
            if len(source) == 0:
                return torch.tensor([self.BOS_TOKEN, self.EOS_TOKEN], dtype=torch.long)
            
            if isinstance(source[0], dict):
                tokens = self.encode_midi_events(
                    source, add_bos=add_bos, add_eos=add_eos, max_length=max_length
                )
            else:
                tokens = [int(t) for t in source]
            
            return torch.tensor(tokens, dtype=torch.long)
        
        raise ValueError(f"Unsupported source type: {type(source)}")
    
    def process_batch(
        self,
        sources: List[Union[List[Dict], List[int], Tensor]],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        max_len = max_length or self.config.max_seq_len
        tensors = []
        
        for source in sources:
            tensor = self.process(source, add_bos, add_eos, max_len)
            tensors.append(tensor)
        
        batch_size = len(tensors)
        lengths = torch.tensor([t.shape[0] for t in tensors])
        max_len_batch = min(lengths.max().item(), max_len)
        
        padded = torch.full(
            (batch_size, max_len_batch),
            self.config.pad_token_id,
            dtype=torch.long,
        )
        
        for i, tensor in enumerate(tensors):
            length = min(tensor.shape[0], max_len_batch)
            padded[i, :length] = tensor[:length]
        
        mask = torch.arange(max_len_batch).unsqueeze(0) >= lengths.unsqueeze(1)
        
        return padded, mask
