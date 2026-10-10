# -*- coding: utf-8 -*-
"""把仓库根目录加进 sys.path，使 `import core.*` 在 pytest 下可用。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
