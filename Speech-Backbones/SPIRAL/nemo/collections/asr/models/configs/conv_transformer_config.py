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
from typing import List, Optional

from omegaconf import MISSING

from nemo.collections.asr.models.configs.common_config import Conv2dBlock, Conv1dNormAct

__all__ = ['AdaptiveFFN', 'RelTransformerBlock', 'ConvTransformerBlock', 'ConvTransformerEncoder']


@dataclass
class AdaptiveFFN:
    gate_mix_prob: float = MISSING
    gate_sample_prob: float = MISSING
    identity_threshold: float = MISSING
    identity_loss_weight: float = MISSING
    init_identiy_bias: float = 0.0
    ffn_residual: bool = True
    norm_in_ffn: bool = True


@dataclass
class RelTransformerBlock:
    n_layer: int = MISSING
    d_model: int = MISSING
    n_head: int = MISSING
    d_head: int = MISSING
    d_inner: int = MISSING
    dropout: float = MISSING
    dropout_att: float = MISSING
    pre_lnorm: bool = False
    norm_output: bool = False
    uni_attn: bool = True
    norm_type: str = 'ln'
    pos_enc: Optional[str] = 'xl'
    layer_drop: float = 0.0

    adaptive_ffn: Optional[AdaptiveFFN] = None


@dataclass
class ConvTransformerBlock:
    conv_layers: List[Conv1dNormAct]
    transformer_block: Optional[RelTransformerBlock] = MISSING


@dataclass
class ConvTransformerEncoder:
    _target_: str = 'nemo.collections.asr.modules.ConvTransformerEncoder'
    feat_in: int = MISSING
    use_conv_mask: bool = MISSING
    conv2d_block: Conv2dBlock = MISSING
    conv_transformer_blocks: List[ConvTransformerBlock] = MISSING
    use_tf_pad: bool = True
    ln_eps: float = 1e-5
    init_mode: str = 'xavier_uniform'
    bias_init_mode: str = 'zero'
