# Copyright 2025 Santiago Gallego
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

import os

# Global constants for the API.
# This is distinct from the model-specific parameters defined in the `configs/config.yaml` file.

SUPPORTED_IMAGE_FORMATS = {"image/bmp", "image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB.
MAX_SIZE = 600  # Image shape size.

SUPPORTED_MODELS = {"craft_parseq"}
CONFIG_PATH = os.getenv("CONFIG_PATH", "./configs/config.yaml")
