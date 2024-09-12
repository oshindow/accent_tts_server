# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import MISSING

import nemo.core.classes.dataset
from nemo.core.config.modelPT import SchedConfig, OptimConfig

__all__ = ['Conv2dNormAct', 'Conv1dNormAct', 'Conv2dBlock', 'DatasetConfig', 'OptimConfig', 'NovogradParams',
           'WarmupParams', 'WarmupHoldParams', 'PolynomialDecayAnnealingParams', 'PolynomialHoldDecayAnnealingParams',
           'Tokenizer']


@dataclass
class Conv2dNormAct:
    filters: int = MISSING
    kernel_size: Tuple[int, int] = MISSING
    stride: Tuple[int, int] = MISSING
    norm_type: Optional[str] = MISSING
    gn_groups: Optional[int] = None
    act_func: Optional[str] = MISSING
    dilation: Tuple[int, int] = (1, 1)
    dropout: float = 0.0
    padding: str = 'same'
    bias: Optional[bool] = None


@dataclass
class Conv1dNormAct:
    filters: int = MISSING
    kernel_size: Tuple[int] = MISSING
    stride: Tuple[int] = MISSING
    norm_type: Optional[str] = MISSING
    gn_groups: Optional[int] = None
    act_func: Optional[str] = MISSING
    dilation: Tuple[int] = (1,)
    dropout: float = 0.0
    padding: str = 'same'
    bias: Optional[bool] = None


@dataclass
class ProjUpsampling:
    rate: int = MISSING
    filters: int = MISSING
    kernel_size: Tuple[int] = MISSING
    norm_type: Optional[str] = MISSING
    act_func: Optional[str] = MISSING
    dropout: float = 0.0
    padding: str = 'same'
    bias: bool = True


@dataclass
class Conv2dBlock:
    layers: List[Conv2dNormAct] = MISSING
    output_dim: int = MISSING


@dataclass
class DatasetConfig(nemo.core.classes.dataset.DatasetConfig):
    manifest_dir: str = MISSING
    data_dir: str = MISSING
    manifest_filepath: str = MISSING
    sample_rate: int = MISSING
    labels: List[str] = MISSING
    trim_silence: bool = False

    # Optional
    int_values: bool = False
    augmentor: Optional[Dict[str, Any]] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    max_utts: int = 0
    dup_factor: int = 1
    blank_index: int = -1
    unk_index: int = -1
    normalize: bool = False
    trim: bool = True
    load_audio: bool = True
    parser: Optional[str] = 'en'
    parser_add_end_space: bool = False
    add_misc: bool = False
    subword_sampling_nbest_size: Optional[int] = None
    subword_sampling_alpha: Optional[float] = None


@dataclass
class AudioDatasetConfig(nemo.core.classes.dataset.DatasetConfig):
    manifest_dir: str = MISSING
    data_dir: str = MISSING
    manifest_filepath: str = MISSING

    sample_rate: int = MISSING

    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    crop_size: Optional[int] = None


@dataclass
class AdamWParams(OptimConfig):
    name: str = 'adamw'

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class NovogradParams(OptimConfig):
    name: str = 'novograd'

    betas: Tuple[float, float] = (0.95, 0.98)
    eps: float = 1e-8
    eps_in_sqrt: bool = False
    weight_decay: float = 0
    weight_decay_ema: bool = True
    grad_averaging: bool = False
    amsgrad: bool = False
    luc: bool = False
    luc_grad_trust: float = 0.0
    luc_grad_trust_rel: bool = False
    luc_trust: float = 1e-3
    luc_trust_min: float = 0.0
    luc_eps: float = 1e-8
    luc_update_min: float = 1e-7
    luc_update_max: float = 1.0


@dataclass
class WarmupParams:
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    warmup_power: Optional[float] = None


@dataclass
class WarmupHoldParams:
    hold_steps: Optional[int] = None
    hold_ratio: Optional[float] = None


@dataclass
class PolynomialDecayAnnealingParams(SchedConfig, WarmupParams):
    name: str = 'PolynomialDecayAnnealing'
    max_steps: int = MISSING
    power: float = 1.0
    cycle: bool = False


@dataclass
class PolynomialHoldDecayAnnealingParams(PolynomialDecayAnnealingParams, WarmupHoldParams):
    name: str = 'PolynomialHoldDecayAnnealing'


@dataclass
class CosineAnnealingParams(SchedConfig, WarmupParams):
    name: str = 'CosineAnnealing'
    max_steps: int = MISSING


@dataclass
class Tokenizer:
    dir: str = MISSING  # path to directory which contains either tokenizer.model (bpe) or vocab.txt (for wpe)
    file: Optional[str] = None
    type: str = 'bpe'
    prepend_unk_to_vocab: bool = True
