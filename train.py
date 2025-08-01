# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

# # Uncomment the following lines to enable debug timing
# timing_options.active = True
# timing_options.print_enabled = True


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "Unknown"


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    logger.info(f"Git hash: {get_git_revision_hash()}")
    logger.info(f"Compiling native code..")
    from threedgrut.trainer import Trainer3DGRUT

    # # NOTE: It is also possible to directly instantiate a trainer from a checkpoint/INGP/PLY file
    # c = OmegaConf.load("example.yaml")
    # trainer = Trainer3DGRUT.create_from_checkpoint("runs/bonsai_3dgrt/bonsai-1806_140944/ours_7000/ckpt_7000.pt", conf)
    # trainer = Trainer3DGRUT.create_from_ingp("export_last.ingp", DictConfig(c))
    # trainer = Trainer3DGRUT.create_from_ply("export_last.ply", DictConfig(c))

    trainer = Trainer3DGRUT(conf)
    trainer.run_training()


if __name__ == "__main__":
    main()
