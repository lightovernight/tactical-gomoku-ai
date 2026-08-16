# Tactical Proof Gomoku AI

一个可直接游玩的 15×15 无禁手五子棋 AI。项目使用小型 Policy-Value
残差网络引导 PUCT，并在搜索树的每个新节点执行严格的一步战术证明，将
已证明的胜负结果向树根传播。

当前仓库是**推理发布版**：包含游戏规则、神经网络、战术证明、批量 MCTS、
Pygame 界面和最终模型；不包含训练器、replay buffer 或训练实验数据。

## 主要特性

- AlphaZero 风格的策略价值网络与 PUCT 搜索
- 全树一步 WIN/LOSS 战术证明
- 已证明胜着优先、已证明败着剪枝和证明结果向上传播
- GPU 批量叶节点推理、虚拟访问计数、叶节点去重与搜索树复用
- 15×15 自由规则：无禁手，五连或长连均获胜
- 可切换先后手的 Pygame 人机界面

## 环境与依赖

推荐使用 Python 3.11。项目锁定并验证了以下运行依赖：

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| Python | `>=3.11,<3.13` | 运行环境 |
| PyTorch | `2.8.0` | Policy-Value 网络和 GPU 推理 |
| NumPy | `2.4.6` | 棋盘、战术分析和 MCTS 数据结构 |
| Pygame | `2.6.1` | 图形界面 |

GPU不是必需条件；程序会优先使用CUDA，没有可用CUDA时自动退回CPU。
发布模型在 Windows、Python 3.11.11、PyTorch 2.8.0+cu128 上验证通过。
`uv`在Windows和Linux上默认使用PyTorch官方CUDA 12.8索引，在macOS上回退
到PyPI的CPU构建。CUDA构建即使没有NVIDIA GPU也可以使用CPU运行，但首次
安装的下载和解包体积较大，后续会复用本地缓存。

依赖以标准 `pyproject.toml` 为唯一声明源，`uv.lock` 固定完整解析结果。
因此推荐使用 `uv`，同时完全兼容普通 `pip`。

## 使用 uv（推荐）

```powershell
python -m pip install uv
uv sync
uv run gomoku-ai
```

也可以使用模块入口：

```powershell
uv run python -m gomoku_ai
```

检查PyTorch是否识别到GPU：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## 使用 pip

普通 `pip` 不读取 `tool.uv.sources`，因此Windows下直接安装得到的是CPU版
PyTorch。CPU安装方式如下：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
gomoku-ai
```

需要CUDA 12.8时，先从PyTorch官方索引安装GPU构建，再安装本项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
gomoku-ai
```

Linux或macOS激活虚拟环境时使用 `source .venv/bin/activate`。

## 游戏参数

默认人类执黑，AI每步执行6400次模拟，叶节点网络推理batch为64：

```powershell
uv run gomoku-ai
```

人类执白：

```powershell
uv run gomoku-ai --human white
```

降低搜索量以加快响应：

```powershell
uv run gomoku-ai --simulations 800 --inference-batch-size 64
```

界面按键：

- `B` / `W`：切换人类执黑或执白并重新开始
- `R`：重新开始
- `Esc` / `Q`：退出

`--inference-batch-size`只控制一次送入网络的叶节点数量，不会改变总模拟
次数。默认64并不是训练batch size。

## 模型信息

- checkpoint：25,000局自我对弈，250,000次梯度更新
- 输入：当前行动方、对手、空位，共 `3×15×15`
- 主干：64通道、4个残差块
- 输出：225维策略logits和一个 `[-1, 1]` 价值
- 参数量：326,403
- 默认模型：`src/gomoku_ai/models/gomoku_25000.pt`
- SHA-256：`7ce7a2f0ab8a9b49e2be0349041acce61c60cf732a0d1437996f7e40203e91da`

进一步说明见 [模型卡](docs/MODEL_CARD.md) 和
[算法说明](docs/ALGORITHM.md)。

## 开发与验证

```powershell
uv sync --group dev
uv run pytest
uv run ruff check .
```

## 项目结构

```text
.
├── pyproject.toml
├── uv.lock
├── src/gomoku_ai/
│   ├── game.py
│   ├── model.py
│   ├── tactics.py
│   ├── mcts.py
│   ├── play.py
│   └── models/gomoku_25000.pt
├── tests/
└── docs/
```

## 已知限制

- 发布版不能从零复现训练过程或现有消融实验。
- 战术求解器精确处理一步胜、唯一防守和多个立即杀点，但不是完整VCF/VCT
  求解器；更深证明来自搜索树中的逐层传播。
- 当前规则是自由五子棋，不能直接用于带禁手的连珠规则。
- AI搜索在GUI主线程执行，高搜索量时窗口可能短暂显示为未响应。
