# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from .tts import GradTTS
from .tts_gstloss import GradTTSGST
from .tts_conformer import GradTTSConformer
from .tts_conformer_gstloss import GradTTSConformerGST
from .tts_conformer_gstloss_grl import GradTTSConformerGSTGRL
from .tts_ori import GradTTSORI
from .tts_conformer_gstloss_grl_focalloss import GradTTSConformerGSTGRLFL