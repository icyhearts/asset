## 2026-07-09 lm-eval + SGLang DeepSeek-V2-Lite-Chat 启动失败原因

结论：这次失败不是因为 `sglang` 源码里没有 `DeepseekV2ForCausalLM` 类。类确实存在，位置是：

```text
/share/users/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py
```

并且文件末尾确实注册了：

```python
EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

真正原因是：SGLang 的模型注册器是通过动态 import 扫描 `sglang.srt.models.*`。`deepseek_v2.py` 在 import 阶段失败了，所以它的 `EntryClass` 根本没有机会注册到 `ModelRegistry`。后面 SGLang 查不到 native `DeepseekV2ForCausalLM`，才 fallback 到 Transformers backend。

日志里的关键证据是：

```text
Ignore import error when loading sglang.srt.models.deepseek_v2: No module named 'cutlass'
DeepseekV2ForCausalLM has no SGLang implementation, falling back to Transformers implementation.
Using Transformers backend.
KeyError: 'sglang'
```

我在同一个 conda 环境里直接验证：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import traceback
try:
    import sglang.srt.models.deepseek_v2 as m
    print("import ok", m.EntryClass)
except Exception:
    traceback.print_exc()
PY
```

得到的直接异常链路是：

```text
sglang.srt.models.deepseek_v2
  -> sglang.srt.layers.attention.dsa.dsa_indexer
  -> sglang.jit_kernel.dsa.__init__
  -> sglang.jit_kernel.dsa.cutedsl_paged_mqa_logits
  -> import cutlass
ModuleNotFoundError: No module named 'cutlass'
```

也就是说，`DeepSeek-V2-Lite-Chat` 本身不是 DSA 模型，命令里也显式用了 `attention_backend="triton"`，但 `deepseek_v2.py` 顶层 import 了 DSA indexer；DSA indexer 又通过 `sglang.jit_kernel.dsa.__init__` eager import 了 CuTe DSL paged MQA logits；这个 CuTe DSL 文件顶层需要 `cutlass`。只要这个依赖缺失或 API 不匹配，整个 `deepseek_v2.py` import 就失败，DeepSeek V2 native 类就不会注册。

为什么会出现 `DeepseekV2ForCausalLM has no SGLang implementation`

SGLang 的注册逻辑在 `python/sglang/srt/models/registry.py`：

```python
ModelRegistry.register("sglang.srt.models")
```

注册器会遍历 `sglang.srt.models` 下的模块，逐个 `importlib.import_module(name)`。如果某个模块 import 抛异常，默认不是 hard fail，而是：

```python
logger.warning(f"Ignore import error when loading {name}: {e}")
continue
```

因此 `deepseek_v2.py` 失败后，`DeepseekV2ForCausalLM` 没有进入 `ModelRegistry.models`。我用同一环境验证过：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from sglang.srt.models.registry import ModelRegistry
print("DeepseekV2ForCausalLM supported:", "DeepseekV2ForCausalLM" in ModelRegistry.get_supported_archs())
print("DeepseekV3ForCausalLM supported:", "DeepseekV3ForCausalLM" in ModelRegistry.get_supported_archs())
print("TransformersForCausalLM supported:", "TransformersForCausalLM" in ModelRegistry.get_supported_archs())
PY
```

结果是：

```text
DeepseekV2ForCausalLM supported: False
DeepseekV3ForCausalLM supported: False
TransformersForCausalLM supported: True
```

模型 config 里写的是：

```text
architectures: ['DeepseekV2ForCausalLM']
model_type: deepseek_v2
auto_map: {'AutoConfig': ..., 'AutoModel': ..., 'AutoModelForCausalLM': ...}
```

`get_model_architecture()` 看到 native registry 里没有 `DeepseekV2ForCausalLM`，就会走 Transformers fallback。warning 文案来自 `python/sglang/srt/model_loader/utils.py`：

```python
"%s has no SGLang implementation, falling back to Transformers implementation..."
```

这句话容易误导。这里的“has no SGLang implementation”不是说源码中没有这个类，而是“当前进程的 ModelRegistry 里没有成功注册这个 architecture”。

为什么 fallback 后又 `KeyError: 'sglang'`

fallback 进了 SGLang 的通用 Transformers backend，见日志：

```text
Using Transformers backend.
```

`python/sglang/srt/models/transformers.py` 里会把 HF config 的 attention 实现改成：

```python
self.text_config._attn_implementation = "sglang"
AutoModel.from_config(...)
```

但是这个 DeepSeek-V2-Lite-Chat 的 remote modeling 文件里，`ATTENTION_CLASSES` 只有：

```python
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
}
```

没有 `"sglang"` 这个 key。所以 Transformers remote code 初始化 decoder layer 时：

```python
self.self_attn = ATTENTION_CLASSES[config._attn_implementation](...)
```

就抛了：

```text
KeyError: 'sglang'
```

所以 `KeyError: 'sglang'` 是 fallback 到 Transformers backend 之后的二次错误；根因仍然是前面的 native DeepSeek V2 import/registration 被 `cutlass` 依赖问题打断。

关于当前 conda 环境里的 cutlass 状态

环境里不是完全没装相关包：

```text
nvidia-cutlass-dsl           4.5.2
nvidia-cutlass-dsl-libs-base 4.5.2
nvidia-cutlass-dsl-libs-cu13 4.5.2
```

但直接检查：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import importlib.util
print(importlib.util.find_spec("cutlass"))
PY
```

结果是：

```text
None
```

`nvidia-cutlass-dsl` 的 wheel 把 Python 包放在：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass
```

但这个目录没有自动进入 `sys.path`，所以普通 `import cutlass` 找不到。

即使手动加：

```bash
export PYTHONPATH=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages:$PYTHONPATH
```

新的错误也会变成：

```text
ImportError: cannot import name 'HardwareInfo' from 'cutlass.utils'
```

说明当前源码中的：

```python
from cutlass.utils import HardwareInfo
```

和这个环境里 `nvidia-cutlass-dsl==4.5.2` 实际提供的 Python API 不匹配。单纯加 `PYTHONPATH` 不是完整修复。

建议修复方式

优先推荐代码修复：不要让非必需的 CuTe DSL DSA kernel 在模型注册阶段 eager import。对这次命令来说，模型是 DeepSeek V2 Lite，attention backend 是 `triton`，GPU 是 H100/SM90；`cutedsl_paged_mqa_logits.py` 主要是 Blackwell/SM100 的 DSA paged MQA logits 可选路径，不应该因为它的依赖问题导致整个 `DeepseekV2ForCausalLM` native 实现注册失败。

最小修复方向：

1. 修改 `python/sglang/jit_kernel/dsa/__init__.py`，不要在 package import 时直接 import `.cutedsl_paged_mqa_logits`。
2. 把 `CuteDSLPagedMQALogitsRunner` 和 `pick_dsl_expand` 变成懒加载，或者在 `ImportError` 时降级为 `None`。
3. `cutedsl_paged_mqa_logits()` 函数本身已经在 `python/sglang/jit_kernel/dsa/paged_mqa_logits.py` 内部懒 import `CuteDSLPagedMQALogitsRunner`，所以 common path 不需要在 `__init__.py` 里提前 import 它。
4. `dsa_indexer.py` 里已有类似保护逻辑：`pick_dsl_expand = None`，只是在当前代码中顶部那次 `from sglang.jit_kernel.dsa import ...` 已经先触发失败了。应让这段保护真正生效。

示意补丁：

```python
# python/sglang/jit_kernel/dsa/__init__.py
from sglang.srt.utils import is_hip

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

CuteDSLPagedMQALogitsRunner = None
pick_dsl_expand = None

if not is_hip():
    try:
        from .cutedsl_paged_mqa_logits import (
            CuteDSLPagedMQALogitsRunner,
            pick_dsl_expand,
        )
    except ImportError:
        # CuTe DSL is optional. Do not break model registration when unavailable.
        pass
```

更稳妥的版本是连这个 try import 都不要放在 `__init__.py`，而是让使用 CuTe DSL 的具体路径在 runtime import；当用户真的设置 `--dsa-paged-mqa-logits-backend cutedsl` 且依赖不可用时，再抛明确错误。

如果只想先让这条 lm-eval 命令跑通，修完后用下面命令验证 registry：

```bash
cd /share/users/like/package/sglang_kernel_src

/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from sglang.srt.models.registry import ModelRegistry
print("DeepseekV2ForCausalLM supported:", "DeepseekV2ForCausalLM" in ModelRegistry.get_supported_archs())
print("DeepseekV3ForCausalLM supported:", "DeepseekV3ForCausalLM" in ModelRegistry.get_supported_archs())
PY
```

修复成功后应看到：

```text
DeepseekV2ForCausalLM supported: True
DeepseekV3ForCausalLM supported: True
```

再验证直接 import：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sglang.srt.models.deepseek_v2 as m
print([cls.__name__ for cls in m.EntryClass])
PY
```

预期：

```text
['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepseekV32ForCausalLM']
```

环境修复方向也可以做，但不如代码修复稳。需要让当前源码期望的 `import cutlass`、`import cutlass.cute`、`from cutlass.utils import HardwareInfo` 都成立。当前 `nvidia-cutlass-dsl==4.5.2` 没有自动暴露 `cutlass` 顶层包，并且加 path 后也没有 `HardwareInfo`，所以需要找到和这份 SGLang 源码匹配的 CuTe DSL/CUTLASS Python package 版本，而不是只确认 `pip show nvidia-cutlass-dsl` 存在。

不建议把这次问题通过 `--model_impl transformers` 或类似方式绕过去。因为当前 HF remote DeepSeek V2 code 的 `ATTENTION_CLASSES` 不支持 `"sglang"`，fallback backend 本身也已经证明会 `KeyError: 'sglang'`。正确方向是恢复 native SGLang `DeepseekV2ForCausalLM` 注册。

最终根因链路可以概括为：

```text
lm-eval 启动 sglang
  -> SGLang 扫描并注册 sglang.srt.models
  -> import sglang.srt.models.deepseek_v2
  -> 顶层 import DSA indexer
  -> DSA package __init__ eager import CuTe DSL paged MQA logits
  -> cutlass / HardwareInfo 依赖不可用或 API 不匹配
  -> deepseek_v2.py import 失败
  -> DeepseekV2ForCausalLM 没有注册进 ModelRegistry
  -> get_model_architecture 认为没有 native SGLang implementation
  -> fallback 到 TransformersForCausalLM
  -> Transformers backend 设置 _attn_implementation="sglang"
  -> DeepSeek remote modeling 只有 eager/flash_attention_2，没有 sglang
  -> KeyError: 'sglang'
```

## 2026-07-09 simo_sglang 环境无法 import cutlass 的原因和修复

现象不是简单的 Python 版本问题，而是 `nvidia-cutlass-dsl` 这个包在两个环境里的实际安装内容不同。

在可以工作的环境 `/share_data/users/like/miniconda3/envs/simo_sglang_pip/` 里，`site-packages` 下有：

```text
nvidia_cutlass_dsl.pth
nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py
nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py
nvidia_cutlass_dsl/dsl_packages/cutlass/...
```

其中 `nvidia_cutlass_dsl.pth` 的内容是：

```text
nvidia_cutlass_dsl/python_packages
```

Python 启动时会读取 `.pth` 文件，把这个目录加入 `sys.path`，所以顶层包 `cutlass` 能被找到。实际 import 路径是：

```text
cutlass file: /share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
cute file: /share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py
HardwareInfo: <class 'cutlass.utils.hardware_info.HardwareInfo'>
```

但在不能工作的环境 `/share_data/users/like/miniconda3/envs/simo_sglang/` 里，检查到：

```text
site-packages/nvidia_cutlass_dsl.pth 不存在
nvidia_cutlass_dsl/include/CuteDSLRuntime.h
nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so
```

也就是说，这个环境里虽然 `pip show nvidia-cutlass-dsl` 会显示 `4.5.2` 已安装，但 `site-packages/nvidia_cutlass_dsl/` 下面缺少真正提供 Python 顶层模块的 `python_packages/cutlass/...`，也缺少负责把它加入 `sys.path` 的 `.pth` 文件。因此：

```python
import cutlass
```

会直接 `ModuleNotFoundError: No module named 'cutlass'`。

这也解释了为什么 `pip show` 看起来一样但行为不同：`pip show` 主要读的是 `*.dist-info/METADATA`，它只能说明包元数据存在，不能保证包的 Python payload、`.pth` 文件和依赖文件都完整存在。

更关键的是，SGLang 当前源码里 `python/sglang/jit_kernel/dsa/cutedsl_paged_mqa_logits.py` 需要的不只是 `import cutlass`，还包括：

```python
import cutlass.cute as cute
from cutlass.utils import HardwareInfo
```

所以只手动加一个 `PYTHONPATH=.../nvidia_cutlass_dsl/python_packages` 不一定够。如果目标环境里的 `nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py` 本身不存在，后面仍然会失败。

推荐修复方式是重装目标环境里的 CUTLASS DSL 相关包，让它重新落下完整文件：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu13 \
  nvidia-cutlass-dsl-libs-core

/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pip install --no-cache-dir --force-reinstall \
  nvidia-cutlass-dsl==4.5.2 \
  nvidia-cutlass-dsl-libs-base==4.5.2 \
  nvidia-cutlass-dsl-libs-cu13==4.5.2
```

重装后先确认这些文件存在：

```bash
SITE=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages

ls -l "$SITE/nvidia_cutlass_dsl.pth"
ls -l "$SITE/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py"
ls -l "$SITE/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py"
ls -l "$SITE/nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py"
```

再验证 import：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sys
import cutlass
import cutlass.cute
from cutlass.utils import HardwareInfo

print("cutlass:", cutlass.__file__)
print("cutlass.cute:", cutlass.cute.__file__)
print("HardwareInfo:", HardwareInfo)
print("has cutlass path:", [p for p in sys.path if "nvidia_cutlass_dsl" in p])
PY
```

预期能看到 `cutlass` 来自：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
```

如果重装后仍然缺文件，说明当前 pip index/cache 里拿到的 wheel 有问题，或者安装过程中被其他包覆盖。因为 `/share_data/users/like/miniconda3/envs/simo_sglang_pip/` 已经有一份可工作的安装，可以用它作为对照。临时且确定的同步方式是只同步 `nvidia_cutlass_dsl` 这组文件：

```bash
SRC=/share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages
DST=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages

cp -a "$SRC/nvidia_cutlass_dsl.pth" "$DST/"
cp -a "$SRC/nvidia_cutlass_dsl" "$DST/nvidia_cutlass_dsl.from_pip_env"
mv "$DST/nvidia_cutlass_dsl" "$DST/nvidia_cutlass_dsl.broken.$(date +%Y%m%d_%H%M%S)"
mv "$DST/nvidia_cutlass_dsl.from_pip_env" "$DST/nvidia_cutlass_dsl"
```

如果还想让 `pip show` 的元数据也和工作环境一致，可以再同步 dist-info：

```bash
SRC=/share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages
DST=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages

cp -a "$SRC/nvidia_cutlass_dsl-4.5.2.dist-info" "$DST/nvidia_cutlass_dsl-4.5.2.dist-info.from_pip_env"
mv "$DST/nvidia_cutlass_dsl-4.5.2.dist-info" "$DST/nvidia_cutlass_dsl-4.5.2.dist-info.broken.$(date +%Y%m%d_%H%M%S)"
mv "$DST/nvidia_cutlass_dsl-4.5.2.dist-info.from_pip_env" "$DST/nvidia_cutlass_dsl-4.5.2.dist-info"
```

同步完成后，重新开一个 shell 或重新运行 Python，再执行上面的 import 验证。

最后还要验证 SGLang 的 DeepSeek V2 注册是否恢复：

```bash
cd /share/users/like/package/sglang_kernel_src

/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sglang.srt.models.deepseek_v2 as m
from sglang.srt.models.registry import ModelRegistry

print([cls.__name__ for cls in m.EntryClass])
print("DeepseekV2ForCausalLM supported:", "DeepseekV2ForCausalLM" in ModelRegistry.get_supported_archs())
PY
```

预期：

```text
['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepseekV32ForCausalLM']
DeepseekV2ForCausalLM supported: True
```

如果这里只修 `cutlass` 后 DeepSeek V2 注册恢复，那么之前日志里的：

```text
DeepseekV2ForCausalLM has no SGLang implementation, falling back to Transformers implementation
```

也会随之消失。因为那条日志的直接原因不是源码里没有 `DeepseekV2ForCausalLM`，而是导入 `deepseek_v2.py` 的过程中被缺失的 `cutlass` 依赖打断，导致模型类没有注册进 `ModelRegistry`。

## 2026-07-09 为什么 sglang editable 依赖安装出的 nvidia-cutlass-dsl 不完整

先看结论：这不是因为 SGLang 的依赖少写了 `[cu13]`。`/share/users/like/package/sglang_kernel_src/python/pyproject.toml` 里已经写的是：

```toml
"nvidia-cutlass-dsl[cu13]==4.5.2",
```

`/share/users/like/package/sglang_kernel_src/like-useful/install-sglang.sh` 做的是：

```bash
pip install --config-settings=build.verbose=true -vvv -e "python" --no-build-isolation
```

所以按依赖声明来说，安装 SGLang 时带入的 CUTLASS DSL 和手动执行下面命令，解析出来的依赖闭包应当是同一组：

```bash
pip install "nvidia-cutlass-dsl[cu13]==4.5.2"
```

真正的差异来自安装时的环境状态：SGLang editable 安装是在一个已经有旧版 `nvidia-cutlass-dsl 4.3.5` 的环境里做升级；你手动修复时先进入 `site-packages` 删除了 `*cutlass*`，相当于做了 clean install。

从安装日志 `/share/users/like/package/sglang_kernel_src/temp/pip-sglang-log.main-local-dep.txt.2026_07_09___11_39_02` 可以看到关键顺序：

```text
27072 Installing collected packages: ..., nvidia-cutlass-dsl-libs-base,
      ..., nvidia-cutlass-dsl-libs-cu13, nvidia-cutlass-dsl, ...

27232 Attempting uninstall: nvidia-cutlass-dsl
27233 Found existing installation: nvidia-cutlass-dsl 4.3.5
27238 Removing file or directory .../site-packages/nvidia_cutlass_dsl.pth
27242 Removing file or directory .../nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
27384 Removing file or directory .../nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py
27452 Removing file or directory .../nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py
27461 Successfully uninstalled nvidia-cutlass-dsl-4.3.5

27525 Successfully installed ... nvidia-cutlass-dsl-4.5.2
      nvidia-cutlass-dsl-libs-base-4.5.2
      nvidia-cutlass-dsl-libs-cu13-4.5.2 ...
```

这里的问题是 `nvidia-cutlass-dsl` 的打包方式发生了变化。4.5.2 里 `nvidia-cutlass-dsl` 本体更像一个 metadata/meta package，它自己的 `RECORD` 只有少量 dist-info 文件。真正的 Python 文件，例如：

```text
nvidia_cutlass_dsl.pth
nvidia_cutlass_dsl/python_packages/cutlass/__init__.py
nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py
nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py
```

是在 `nvidia-cutlass-dsl-libs-base==4.5.2` 和 `nvidia-cutlass-dsl-libs-cu13==4.5.2` 这两个 wheel 的 `RECORD` 里。

但是旧的 `nvidia-cutlass-dsl==4.3.5` 也记录并拥有这些同名路径。pip 在同一个大事务里升级时，先把 4.5.2 的 `libs-base`、`libs-cu13` 文件落到同样的 `nvidia_cutlass_dsl/...` 路径，然后卸载旧的 `nvidia-cutlass-dsl 4.3.5`。旧包卸载是按旧 wheel 的 `RECORD` 删除文件，它不知道这些路径刚刚又被 4.5.2 的 split wheels 写过，于是把新包需要的文件也删掉了。

这就是为什么最后会出现非常迷惑的状态：

```text
pip show nvidia-cutlass-dsl              显示 4.5.2 已安装
pip show nvidia-cutlass-dsl-libs-base    显示 4.5.2 已安装
pip show nvidia-cutlass-dsl-libs-cu13    显示 4.5.2 已安装

但是 site-packages/nvidia_cutlass_dsl.pth 不存在
但是 python_packages/cutlass/... 不完整
但是 import cutlass 失败
```

`pip show` 只说明 dist-info 元数据在，不说明 wheel 的 payload 文件仍然完整。

你手动执行下面流程能修好，原因也正是这个：

```bash
cd /share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/
rm -rf *cutlass*
pip install "nvidia-cutlass-dsl[cu13]==4.5.2"
```

这次没有旧的 `nvidia-cutlass-dsl 4.3.5` 在安装中途卸载，也就不会把新装的 `nvidia_cutlass_dsl.pth`、`cutlass/__init__.py`、`cutlass.cute`、`cutlass.utils.HardwareInfo` 删除掉，所以安装结果是完整的。

SGLang 自己的 CI 脚本其实也意识到了 CUTLASS DSL 这组 wheel 有“多个包写同一路径”的问题。`scripts/ci/cuda/ci_install_dependency.sh` 里有：

```bash
force_reinstall_cutlass_dsl_libs_cu13() {
    # nvidia-cutlass-dsl[cu13] has additive PyPI extras: installing it pulls in
    # both -libs-base and -libs-cu13. The two wheels ship intentionally-different
    # content for the same paths ...
    ...
    $PIP_CMD install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}" $PIP_INSTALL_SUFFIX
}
```

这个函数处理的是另一个相近风险：`libs-base` 和 `libs-cu13` 也会写同一些路径，最后必须让 `libs-cu13` 再装一遍，保证 `.py` 和 `.so` 都来自 cu13 wheel。你的 `like-useful/install-sglang.sh` 没有做这一步，也没有在安装前清掉旧 CUTLASS DSL，所以更容易留下不完整状态。

后续建议把安装脚本改成“安装前清理旧 CUTLASS DSL，安装后再 force reinstall cu13 payload”。例如：

```bash
set -x
LOG=temp/pip-sglang-log.main-local-dep.txt.`nowstr.sh`
source like-useful/env-build-pip.sh

SITE=$(
python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu13 \
  nvidia-cutlass-dsl-libs-core || true

rm -rf \
  "$SITE"/nvidia_cutlass_dsl \
  "$SITE"/nvidia_cutlass_dsl.pth \
  "$SITE"/nvidia_cutlass_dsl-*.dist-info \
  "$SITE"/nvidia_cutlass_dsl_libs_base-*.dist-info \
  "$SITE"/nvidia_cutlass_dsl_libs_cu13-*.dist-info \
  "$SITE"/nvidia_cutlass_dsl_libs_core-*.dist-info

pip install --config-settings=build.verbose=true -vvv -e "python" --no-build-isolation > "$LOG" 2>&1

pip install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-cu13==4.5.2"

python - <<'PY'
import cutlass
import cutlass.cute
from cutlass.utils import HardwareInfo
print("cutlass:", cutlass.__file__)
print("cutlass.cute:", cutlass.cute.__file__)
print("HardwareInfo:", HardwareInfo)
PY
```

如果不想改安装脚本，至少在每次 `pip install -e python` 后追加一次：

```bash
pip install --force-reinstall --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2"
pip install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-cu13==4.5.2"
```

更干净的做法是新建 conda env 后第一次安装 SGLang，或者在安装 SGLang 前先卸载并删除旧的 `nvidia_cutlass_dsl*` 文件。只要不是从残留的 `nvidia-cutlass-dsl 4.3.5` 原地升级，就不会触发这次“旧 RECORD 删除新文件”的问题。

## 2026-07-09 让 SIMO sglang_simo 适配 SGLang main 的接口变化

本次修改位于 `/share/users/like/package/simo_conda_sglang`，目标是让 SIMO 基于旧 SGLang 分支实现的自定义 KV cache quantization / attention backend 能在 SGLang `main` 的接口下继续注册和运行。

对比了 SGLang `main-2026_07_08___17_53_37` 和 `main` 后，和 SIMO 相关的主要接口变化有三类：

```text
1. MHATokenToKVPool.__init__ 新增参数：
   kv_cache_layout=None
   post_capture_active=False

2. MHA/MLA KV 写入路径开始支持 KVWriteLoc：
   loc_info 可能不再是裸 Tensor，而是 KVWriteLoc(loc, swa_loc, full_loc)
   用来承载 SWA/unified memory 下的不同写入目标。

3. Triton attention wrapper 新增 page_size：
   decode_attention_fwd(..., page_size=...)
   extend_attention_fwd(..., page_size=...)
   main 分支的 kernel 已经支持 page-aware KV buffer。
```

SIMO 当前的 quantized KV cache buffer 仍然是 3D NHD layout：

```text
k_buffer: [size + page_size, head_num, packed_head_size + scale_head_size] uint8
v_buffer: [size + page_size, head_num, packed_head_size + scale_head_size] uint8
```

它还没有实现 SGLang main 新增的 page-major/unified/post-capture KV 物理布局。因此这次适配选择保守策略：兼容 main 的参数和对象类型，但对 SIMO 尚未支持的 KV layout 快速报错，避免 silent wrong write。

修改点如下。

### 1. init_memory_pool_patch.py

文件：

```text
simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py
```

`SIMOMHATokenToKVPoolAdapter` 增加了 SGLang main 的新参数：

```python
kv_cache_layout=None
post_capture_active=False
```

并且做了限制：

```python
if kv_cache_layout not in (None, "nhd"):
    raise ValueError(...)
if post_capture_active:
    raise ValueError(...)
```

原因是 SIMO 的 quantized KV buffer 目前只实现了 3D NHD 布局，不支持 `PageMajorMHATokenToKVPool` 的 4D/page-major 物理布局，也不支持 main 分支的 post-capture KV sizing。

在 `_patched_init_memory_pool` 里也增加了提前检查：

```python
if server_args.enable_page_major_kv_layout:
    raise ValueError(...)
if server_args.enable_unified_memory:
    raise ValueError(...)
if self.post_capture_kv_active:
    raise ValueError(...)
```

这样如果用户在 SIMO KV cache quantization 下打开了 main 新增的 page-major/unified/post-capture 功能，会在初始化阶段明确失败，而不是继续创建错误布局的 quantized KV buffer。

### 2. memory_pool.py

文件：

```text
simo/extensions/sglang_simo/mem_cache/memory_pool.py
```

增加了对 SGLang main `KVWriteLoc` 的兼容：

```python
try:
    from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
except ImportError:
    def unwrap_write_loc(loc_info):
        return loc_info, None, None
```

这样代码同时兼容旧 SGLang 分支和 main 分支。

`SIMOMHATokenToKVPool.__init__` 新增：

```python
kv_cache_layout: Optional[str] = None
post_capture_active: bool = False
```

并在调用父类 `MHATokenToKVPool` 前检查父类签名：只有当前 SGLang 版本的父类真的支持 `kv_cache_layout` / `post_capture_active` 时才传入。这样同一份 SIMO 代码既能接 SGLang main，也不会在旧分支父类不认识这些 kwargs 时 `TypeError`。

`SIMOMHATokenToKVPool.set_kv_buffer` 现在可以接收：

```python
loc
dcp_kv_mask: Optional[torch.Tensor] = None
```

并先解包：

```python
loc, _, full_loc = unwrap_write_loc(loc)
if full_loc is not None:
    loc = full_loc
```

这里 `full_loc` 是 main 分支 unified memory 场景下预翻译后的物理 full-attention 写入位置。虽然 SIMO 当前已经禁止 unified memory 初始化，但这个处理让接口对齐 main 的写入路径，也避免未来在普通 `KVWriteLoc` 下直接把 dataclass 当 Tensor 用。

同时支持 DCP mask：

```python
if dcp_kv_mask is not None:
    loc = loc[dcp_kv_mask]
    cache_k = cache_k[dcp_kv_mask]
    cache_v = cache_v[dcp_kv_mask]
```

main 的 `TritonAttnBackend._set_kv_buffer` 在 DCP 场景会传 `dcp_kv_mask` kwarg；不接这个参数会直接 `TypeError`。

`SIMOMLATokenToKVPool.set_kv_buffer` 也做了同样的 `KVWriteLoc` 解包和 `dcp_kv_mask` 过滤。

### 3. triton_simo_backend.py

文件：

```text
simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

在调用 SIMO 自定义 extend/decode attention wrapper 时，增加：

```python
page_size=self.page_size
```

对应 SGLang main 的 `TritonAttnBackend` 行为。main 分支的 backend 会从 `model_runner.page_size` 读出 page size，并在 attention wrapper 中传给 kernel。

SIMO 当前自定义 kernel 仍按 3D NHD quantized buffer 访问，因此 page-aware 4D layout 没有启用。`page_size=1` 时行为和旧分支一致。

### 4. decode_attention.py / extend_attention.py

文件：

```text
simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py
simo/extensions/sglang_simo/layers/attention/triton_ops/extend_attention.py
```

两个 wrapper 都新增了参数：

```python
page_size: int = 1
```

并增加了 3D buffer 检查：

```python
if k_buffer.ndim != 3 or v_buffer.ndim != 3:
    raise ValueError(
        "SIMO quantized KV attention expects 3-D NHD KV buffers; ..."
    )
```

这和前面的初始化限制是一致的：SIMO quantized KV attention 当前只支持自身分配的 3D uint8 NHD buffer。如果未来要支持 SGLang main 的 page-major/unified KV，需要把 SIMO Triton kernels 的地址计算也迁移到 main 的 page-aware stride 方式。

### 验证

我用 SGLang main 创建了临时 worktree：

```text
/tmp/sglang-main-codex
```

然后用目标 conda 环境运行了以下验证。

1. 编译 SIMO sglang 扩展：

```bash
PYTHONPATH=/tmp/sglang-main-codex/python:/tmp/sglang-main-codex/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m compileall -q simo/extensions/sglang_simo
```

结果：通过。

2. 在 SGLang main 下注册 SIMO extension：

```bash
PYTHONPATH=/tmp/sglang-main-codex/python:/tmp/sglang-main-codex/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sglang
print("sglang:", sglang.__file__)
from simo.extensions.sglang_simo import register_simo_extensions
register_simo_extensions()
print("register ok")
PY
```

结果：

```text
sglang: /tmp/sglang-main-codex/python/sglang/__init__.py
register ok
```

3. 确认 `triton_simo` backend 注册成功：

```bash
PYTHONPATH=/tmp/sglang-main-codex/python:/tmp/sglang-main-codex/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from simo.extensions.sglang_simo import register_simo_extensions
register_simo_extensions()
print("triton_simo registered:", "triton_simo" in ATTENTION_BACKENDS)
print("factory:", ATTENTION_BACKENDS.get("triton_simo"))
PY
```

结果：

```text
triton_simo registered: True
factory: <function create_triton_simo_backend ...>
```

4. 模拟 main 分支新增参数进入 SIMO adapter：

```text
SIMOMHATokenToKVPoolAdapter 可以接收 kv_cache_layout/post_capture_active。
kv_cache_layout='page_major_layer_major' 会明确报错。
post_capture_active=True 会明确报错。
```

5. 旧 SGLang 当前分支兼容性：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sglang
print("sglang:", sglang.__file__)
from simo.extensions.sglang_simo import register_simo_extensions
register_simo_extensions()
print("register old-current ok")
PY
```

结果：

```text
register old-current ok
```

### 当前限制

这次适配让 SIMO 能接上 SGLang main 的接口，但没有实现 main 新增 KV layout 的完整功能。运行 SIMO KV cache quantization 时应避免同时开启：

```text
--enable-page-major-kv-layout
--enable-unified-memory
post-capture KV sizing 相关配置
```

如果后续必须支持这些功能，需要进一步改 SIMO 的 `set_kv_buffer`、decode kernel、extend kernel，把 KV cache 地址计算从当前 3D NHD 方式升级为 main 的 page-aware stride 方式。

## 2026-07-09 关于 triton_backend.py forward_extend 新增 KVWriteLoc/_attn_output 的 SIMO 修改

需要改，尤其是 `KVWriteLoc` 和 `_set_kv_buffer` 这部分。

SGLang main 的 `TritonAttnBackend.forward_extend` 相比旧分支有几个和 SIMO 相关的变化：

```text
1. attn_out = getattr(forward_batch, "_attn_output", None)
   如果 ForwardBatch 里已有预分配 attention output buffer，就复用它。

2. 保存 KV cache 时不再直接传 forward_batch.out_cache_loc，
   而是构造 KVWriteLoc(
       forward_batch.out_cache_loc,
       self.forward_metadata.swa_out_cache_loc,
       full_loc=self.forward_metadata.out_cache_loc_full_physical,
   )

3. 普通写入改走 self._set_kv_buffer(...)
   _set_kv_buffer 会处理 DCP 场景下的 dcp_kv_mask 和本地物理 loc。
```

SIMO 的 `SIMOTritonAttnBackend.forward_extend` 是 override 版本。如果不跟进这些变化，quantized KV cache 路径会绕过 main 的新写入封装。普通非 DCP、非 SWA、非 unified memory 的场景可能还能跑，但 DCP 或后续引入物理写入 loc 的场景会有风险。

本次已修改：

```text
simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

增加兼容导入：

```python
try:
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
except ImportError:
    KVWriteLoc = None
```

增加 helper：

```python
def _make_kv_write_loc(forward_batch, forward_metadata):
    if KVWriteLoc is None:
        return forward_batch.out_cache_loc
    return KVWriteLoc(
        forward_batch.out_cache_loc,
        getattr(forward_metadata, "swa_out_cache_loc", None),
        full_loc=getattr(forward_metadata, "out_cache_loc_full_physical", None),
    )
```

并把 SIMO 的 output 分配逻辑同步到 main：

```python
attn_out = getattr(forward_batch, "_attn_output", None)
if attn_out is not None:
    o = attn_out
elif layer.qk_head_dim != layer.v_head_dim:
    o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
else:
    o = torch.empty_like(q)
```

保存 KV cache 的代码从：

```python
self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
```

改为：

```python
loc_info = _make_kv_write_loc(forward_batch, self.forward_metadata)
if hasattr(self, "_set_kv_buffer"):
    self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
else:
    self.token_to_kv_pool.set_kv_buffer(layer, loc_info, k, v)
```

这里保留 `hasattr(self, "_set_kv_buffer")` 是为了兼容旧 SGLang 分支；旧分支没有这个 helper 时仍然回退到直接调用 pool。

前面已经在 SIMO 的 `memory_pool.py` 中让 `SIMOMHATokenToKVPool.set_kv_buffer` 和 `SIMOMLATokenToKVPool.set_kv_buffer` 支持：

```python
KVWriteLoc / unwrap_write_loc
dcp_kv_mask
```

所以 `forward_extend` 现在可以安全地走 main 的 `_set_kv_buffer` 封装。

验证：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m compileall -q \
  simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py \
  simo/extensions/sglang_simo/mem_cache/memory_pool.py
```

结果：通过。

注册验证：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import sglang
print("sglang:", sglang.__file__)
from simo.extensions.sglang_simo import register_simo_extensions
register_simo_extensions()
from simo.extensions.sglang_simo.layers.attention.triton_simo_backend import KVWriteLoc
print("KVWriteLoc:", KVWriteLoc)
print("register ok")
PY
```

结果：

```text
KVWriteLoc: <class 'sglang.srt.mem_cache.memory_pool.KVWriteLoc'>
register ok
```

## SGLang main 中 `self.dcp_size > 1` 分支什么时候触发，SIMO 是否需要支持

位置：

```text
/share/users/like/package/sglang_kernel_src/python/sglang/srt/layers/attention/triton_backend.py
TritonAttnBackend.forward_extend
```

main 分支里这段逻辑：

```python
if self.dcp_size > 1:
    return self._forward_extend_dcp(
        q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
    )
```

不是由某个模型名字自动触发，而是由运行参数 `dcp_size` 触发。`dcp_size` 是 decode context parallelism size，对应命令行参数：

```bash
--dcp-size N
# 或
--decode-context-parallel-size N
```

只有 `N > 1` 时才会进入这个分支。默认值是 `1`，所以普通启动不会触发。

例如会触发的配置：

```bash
python -m sglang.launch_server \
  --model-path /data/like/hf-models/Llama3.1-8B-Instruct/ \
  --tp-size 2 \
  --dcp-size 2
```

或者在 `lm-eval --model sglang --model_args ...` 里传：

```json
{
  "tp_size": 2,
  "dcp_size": 2
}
```

不会触发的配置：

```json
{
  "tp_size": 1
}
```

你前面的 gsm8k 命令里 `tp_size=1`，并且没有传 `dcp_size`，所以 `dcp_size` 保持默认 `1`，不会进入 `self.dcp_size > 1` 分支。

SGLang main 对 DCP 还有并行约束：

```text
tensor_model_parallel_size % decode_context_parallel_size == 0
```

也就是 `tp_size` 必须能被 `dcp_size` 整除。DCP group 是在每个 TP group 内部创建的。

需要注意：`dcp_size` / `--decode-context-parallel-size` 和 `attn_cp_size` / `--attention-context-parallel-size` 不是同一个参数。这里 `forward_extend` 的 `self.dcp_size > 1` 只对应 decode context parallelism。

对两个目标模型的结论：

```text
/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/
/data/like/hf-models/Llama3.1-8B-Instruct/
```

这两个模型本身都不会自动让 `dcp_size > 1`。只有启动命令显式传了 `--dcp-size N` 或 `--decode-context-parallel-size N`，并且 `N > 1`，才会触发。

所以如果 SIMO 当前目标是支持这两个模型在常规 `tp_size=1` 或未启用 DCP 的推理/评测场景下运行，不需要实现 `_forward_extend_dcp` 分支。

但 SIMO 不能简单复用普通 extend kernel 来假装支持 DCP。main 的 `_forward_extend_dcp` 做了几件额外事情：

```text
1. DCP 下 KV prefix 被切分到不同 DCP rank。
2. 当前 rank 先计算本地 current-token attention。
3. 对 query heads 做 all_gather，让每个 DCP rank 计算自己负责的 prefix KV partial attention。
4. 再用 LSE 做跨 rank 的 attention output 合并。
```

也就是说，DCP 不是多传一个 `page_size` 或 `KVWriteLoc` 就能对齐的接口变化，而是 attention 计算语义发生了变化。SIMO 的自定义 quantized KV cache + 自定义 extend/decode Triton kernel 目前没有实现这套 DCP 分片读写和 LSE merge，所以如果用户显式打开 `dcp_size > 1`，应该直接报错，而不是静默运行。

本次已加保护：

```text
simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py
simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

在 SIMO quantized KV cache 注入阶段，如果发现：

```python
server_args.dcp_size > 1
```

会直接报错：

```text
SIMO quantized KV cache does not support decode context parallelism
(--dcp-size/--decode-context-parallel-size > 1). Run with dcp_size=1.
```

在 `SIMOTritonAttnBackend.__init__` 里也加了同样的兜底检查，避免其他路径绕过 memory pool patch 后继续启动。

最终结论：

```text
1. 当前 SIMO 不需要实现 SGLang main 的 _forward_extend_dcp 分支。
2. 对 DeepSeek-V2-Lite-Chat-16B_A2.4B 和 Llama3.1-8B-Instruct，只要不显式传 dcp_size > 1，就不会触发该分支。
3. SIMO 已显式禁止 quantized KV cache + dcp_size > 1，防止未来误用时产生错误结果。
4. 如果以后确实要支持 DCP，需要单独实现 SIMO 版本的 DCP extend/decode：包括 DCP KV masked write、prefix KV 分片读取、query all_gather、LSE merge，以及 quantized KV upcast kernel 在 DCP 分片下的正确索引。
```

验证：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m compileall -q \
  simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py \
  simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py
```

结果：通过。

注册验证：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from simo.extensions.sglang_simo import register_simo_extensions
register_simo_extensions()
from simo.extensions.sglang_simo.layers.attention.triton_simo_backend import SIMOTritonAttnBackend, KVWriteLoc
print("backend:", SIMOTritonAttnBackend.__name__)
print("KVWriteLoc:", KVWriteLoc)
print("register ok")
PY
```

结果：通过，`KVWriteLoc` 来自 SGLang main 的 `sglang.srt.mem_cache.memory_pool.KVWriteLoc`。

## SGLang main `forward_decode` 的 `_set_kv_buffer(KVWriteLoc, ...)` 变化，SIMO 是否需要改

需要改。

SGLang main 的 `TritonAttnBackend.forward_decode` 在非 MLA 路径下，保存 KV cache 的逻辑已经从直接：

```python
self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
```

改为：

```python
self._set_kv_buffer(
    forward_batch,
    layer,
    KVWriteLoc(
        forward_batch.out_cache_loc,
        self.forward_metadata.swa_out_cache_loc,
        full_loc=self.forward_metadata.out_cache_loc_full_physical,
    ),
    k,
    v,
    layer.k_scale,
    layer.v_scale,
)
```

这个变化和 `forward_extend` 的变化一样，本质上是把 KV 写入位置从一个裸 tensor 升级为 `KVWriteLoc`，并统一通过 `_set_kv_buffer` 处理：

```text
1. 普通物理写入位置：out_cache_loc
2. SWA 写入位置：swa_out_cache_loc
3. full physical 写入位置：out_cache_loc_full_physical
4. DCP 场景下的 masked local write
5. k_scale / v_scale 参数传递
```

SIMO 的 `SIMOTritonAttnBackend.forward_decode` 是 override 版本，如果继续直接调用：

```python
self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
```

就会绕过 SGLang main 的 `_set_kv_buffer` 封装。对当前最普通的 `dcp_size=1`、非 SWA、非 full physical loc 场景可能还能跑，但已经和 main 的接口语义不一致；以后遇到这些新写入位置语义时风险很高。

本次已修改：

```text
simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

在 SIMO 的 `forward_decode` 里同步 main 的写入策略：

```python
if save_kv_cache:
    if getattr(self, "use_mla", self.is_simo_mla_quantized):
        # SIMO MLA KV cache computes and stores its own MX/per-group scales.
        # Do not apply SGLang's per-tensor k_scale before SIMO quantization.
        self.token_to_kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
        )
    else:
        loc_info = _make_kv_write_loc(forward_batch, self.forward_metadata)
        if hasattr(self, "_set_kv_buffer"):
            self._set_kv_buffer(
                forward_batch,
                layer,
                loc_info,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                loc_info,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
```

这里不能完全照搬 SGLang main 的 MLA 分支。

```text
DeepSeek-V2-Lite-Chat-16B_A2.4B 是 MLA 路径。
SGLang main 的原生 MLA KV cache 在写入前可能会执行 k.div_(layer.k_scale)。
但 SIMO MLA KV cache 不是 SGLang 原生 KV cache，它会在自己的写入 kernel 中重新计算 MX/per-group scale。
```

而 Llama3.1-8B-Instruct 是 MHA/GQA 路径，会走 `_set_kv_buffer + KVWriteLoc + k_scale/v_scale`。

### 关于 `k.div_(layer.k_scale)` 是否应该用于 SIMO MLA

不应该用于 SIMO MLA。

`layer.k_scale` 来自 SGLang 的 `BaseKVCacheMethod`。它是原生 KV cache 量化使用的 per-tensor scale，用途是：

```text
1. 写入 SGLang 原生 KV cache 前，对 K/V 做 per-tensor 量化缩放。
2. 从 SGLang 原生 KV cache 读取后，用同一个 scale 做反量化补偿。
```

SIMO 的 KV cache 量化体系不同。以 MLA 写入为例：

```text
SIMOMLATokenToKVPool.set_kv_buffer
  -> set_mla_kv_buffer
    -> concat_and_cache_mla_kernel
```

`concat_and_cache_mla_kernel` 会直接读取传入的 `cache_k_nope/cache_k_rope` 原始浮点值，然后按 SIMO 的 `kv_cache_quant_spec` 计算并写入：

```text
1. MX 格式：packed value + MX scale
2. per-group FP/INT 格式：group quantized value + per-group scale
```

后续 SIMO 的 `decode_attention.py` / `extend_attention.py` 也是从 uint8 KV cache 中读取 SIMO 自己存的 scale，再用 `_unpack_and_dequant_mxfmt` 或 `_dequant_pg_fused` 还原。这个读路径不会额外乘回 `layer.k_scale`。

所以如果在 SIMO MLA 写入前执行：

```python
k.div_(layer.k_scale)
```

效果就是把准备写入 SIMO KV cache 的真实 K 整体缩小了 `layer.k_scale` 倍，而 SIMO 自己的 MX/per-group scale 又会基于缩小后的值重新计算。读出来时只会恢复到缩小后的 K，不会恢复原始 K。

如果 `layer.k_scale != 1.0`，这会导致 attention logits 里的 `QK^T` 比预期少一个全局倍率，结果会错。

因此本次已修正：

```text
simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

SIMO MLA decode 写 KV cache 时不再执行 `k.div_(layer.k_scale)`，而是直接把原始 `k` 传给 `SIMOMLATokenToKVPool.set_kv_buffer`，由 SIMO 写入 kernel 自己完成量化 scale 的计算和存储。

验证：

```bash
PYTHONPATH=/share/users/like/package/sglang_kernel_src/python:/share/users/like/package/sglang_kernel_src/sgl-kernel/python:/share/users/like/package/simo_conda_sglang \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m compileall -q \
  simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py
```

结果：通过。

并用临时 worktree 指向 SGLang `main` 验证 SIMO extension 可以导入：

```text
sglang: /tmp/sglang-main-codex-decode-check/python/sglang/__init__.py
triton_backend: /tmp/sglang-main-codex-decode-check/python/sglang/srt/layers/attention/triton_backend.py
KVWriteLoc: <class 'sglang.srt.mem_cache.memory_pool.KVWriteLoc'>
register ok on sglang main
```

## SGLang main decode_attention.py 的 PAGE_SIZE 和 4D KV cache 触发条件

这次 main 分支在：

```text
/share/users/like/package/sglang_kernel_src/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
```

给 `_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1` 加了 `PAGE_SIZE` 参数。这个 `PAGE_SIZE` 不是模型 config 里的固有维度，而是运行时的 `server_args.page_size`，在 `TritonAttnBackend.__init__` 里从 `model_runner.page_size` 读出来，然后在 decode/extend Triton kernel 调用时传进去。

核心逻辑可以理解为：

```text
PAGE_SIZE == 1:
  kv_loc 直接当成 token slot 下标
  地址 = kv_loc * slot_stride + head * head_stride + dim

PAGE_SIZE > 1:
  kv_loc 先拆成 page 内地址
  page_id  = kv_loc // PAGE_SIZE
  tok_in_p = kv_loc % PAGE_SIZE
  地址 = page_id * page_stride + tok_in_p * tok_stride + head * head_stride + dim
```

所以要真正进入 `decode_attention.py` 里的 `PAGE_SIZE > 1` 分支，需要同时满足两个条件：

```text
1. 当前 attention decode 走 Triton backend，也就是会调用这个 decode_attention.py。
2. 运行时解析出来的 server_args.page_size > 1。
```

如果某个后端把 `page_size` 改成了 64，但 attention backend 本身不是 Triton，例如 `trtllm_mha` / `trtllm_mla`，那通常不会执行这个 `triton_ops/decode_attention.py` 里的 kernel。这个区别很重要。

### page_size > 1 的主要触发方式

最直接的方式是命令行或 `model_args` 显式设置：

```bash
--page-size 16
--page-size 32
--page-size 64
```

如果是 lm-eval 的 sglang model_args，则可以写成：

```json
{"attention_backend": "triton", "page_size": 64}
```

在这个组合下，如果模型 decode 确实走 `TritonAttnBackend`，`decode_attention.py` 的 `PAGE_SIZE > 1` 分支就会被编译并执行。

SGLang main 里还有一些自动改写 `page_size` 的路径：

```text
默认值:
  - CUDA / 非 MUSA / 没有特殊 env: page_size = 1
  - MUSA: page_size = 64
  - ROCm HIP 且 SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d: page_size = 64

后端约束:
  - flashmla: page_size = 64
  - cutlass_mla: page_size = 128
  - trtllm_mla: page_size 只能是 32/64，否则改成 64
  - tokenspeed_mla: page_size 只能是 32/64，否则改成 64
  - cutedsl_mla: page_size 只能是 32/64，否则改成 64
  - trtllm_mha: page_size 只能是 16/32/64，否则改成 64
  - fa4 + non-MLA + SM100: page_size = 128
  - intel_xpu decode backend: MLA 用 16/32/64/128，MHA 用 64/128，否则改成 128

模型族 override:
  - DeepSeek DSA，也就是 DeepseekV3/DeepseekV32/GLM5 等 DSA 路径: page_size = 64
    ROCm 且不能用 preshuffle paged-MQA 时会退回 page_size = 1
  - DeepseekV4ForCausalLM: CUDA 上 page_size = 256，NPU 上 page_size = 128
  - Qwen3.5 hybrid 在部分 SM100 + trtllm_mha 路径: page_size = 64
  - Qwen3VL + HIP + AITER unified attention: page_size = 16
```

但上面这些自动改写不等价于“一定会进入 `decode_attention.py`”。例如 `trtllm_mha` 会把 MHA 的 page size 改成 64，但它对应 TRT-LLM MHA backend，不是 Triton decode kernel。

### 这两个模型会不会触发 PAGE_SIZE > 1

本地两个模型的 config 是：

```text
/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/
  architectures = ["DeepseekV2ForCausalLM"]
  model_type = "deepseek_v2"

/data/like/hf-models/Llama3.1-8B-Instruct/
  architectures = ["LlamaForCausalLM"]
  model_type = "llama"
```

`DeepseekV2ForCausalLM` 在 `ModelConfig` 里会被识别成 MLA attention，但它不属于 main 分支里 `_deepseek_family_overrides` 注册的 DSA 模型列表。那个列表主要是 `DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM`、`GlmMoeDsaForCausalLM` 等。因此 DeepSeek-V2 Lite 这个模型本身不会自动把 `page_size` 改成 64。

`LlamaForCausalLM` 是普通 MHA/GQA 路径，也没有模型级别的 `page_size` override。

所以在常见 CUDA/Hopper 环境里，如果命令没有传 `page_size`，例如你之前常用的：

```json
{"attention_backend": "triton", "...": "..."}
```

并且没有 `"page_size": 64` 之类的参数，那么这两个模型解析后的 `page_size` 都是 1，不会触发 `decode_attention.py` 里的 `PAGE_SIZE > 1` 分支。

会触发的例子是：

```bash
--attention-backend triton --page-size 64
```

或者 lm-eval:

```json
{"attention_backend": "triton", "page_size": 64}
```

这时 DeepSeek-V2 Lite 和 Llama3.1-8B 都可以把 `PAGE_SIZE=64` 传进 Triton decode kernel。DeepSeek-V2 Lite 的 MLA KV buffer 仍然是 3D 的 MLA buffer；Llama3.1 的普通 MHA KV buffer 默认也是 3D，除非额外打开 page-major/unified-memory layout。

如果在 Blackwell/SM100 上不给 Llama 显式指定 backend，SGLang 的默认 MHA backend 可能选择 `trtllm_mha`，并把 `page_size` 约束到 64。但这种情况下重点是 TRT-LLM MHA backend，不是 `triton_ops/decode_attention.py` 这份 Triton kernel。

### _extract_kv_strides 和 4D KV cache buffer

`_extract_kv_strides(buf, page_size)` 是为了让同一份 Triton decode kernel 同时读两种 KV buffer：

```text
3D:
  [max_slots, head_num, head_dim]

4D page-major:
  [num_pages, page_size, head_num, head_dim]
```

对于 3D buffer，它会合成 page stride：

```text
slot_stride = buf.stride(0)
page_stride = slot_stride * page_size
tok_stride  = slot_stride
```

这样即使 `PAGE_SIZE > 1`，地址也满足：

```text
(kv_loc // page_size) * page_stride + (kv_loc % page_size) * tok_stride
== kv_loc * slot_stride
```

也就是说，`page_size > 1` 不要求 KV buffer 必须是 4D。3D buffer 也可以通过这个 stride 关系继续工作。

真正的 4D page-major KV buffer 主要来自：

```bash
--enable-page-major-kv-layout
```

或者：

```bash
--enable-unified-memory
```

因为 `--enable-unified-memory` 会隐式打开 `enable_page_major_kv_layout`。

SGLang main 里 `PageMajorMHATokenToKVPool` 会用一个共享的 raw uint8 buffer 创建每层 K/V 的 strided view：

```text
k_buffer[layer].shape = (num_pages, page_size, head_num, head_dim)
v_buffer[layer].shape = (num_pages, page_size, head_num, v_head_dim)
```

这个 shape 正是 `_extract_kv_strides` 里 4D 分支期待的格式，也就是 dim-1 必须等于 `page_size`。这种 page-major MHA pool 会用于 MHA-shaped 的 full/SWA/hybrid full-attention KV pool。

对两个目标模型分别看：

```text
Llama3.1-8B-Instruct:
  普通 MHA/GQA 模型。
  默认不开 page-major 时，KV buffer 是 3D。
  如果加 --enable-page-major-kv-layout 且 attention backend 是 triton，
  就会走 PageMajorMHATokenToKVPool，KV buffer 变成 4D page-major view。

DeepSeek-V2-Lite-Chat-16B_A2.4B:
  MLA 模型。
  普通路径使用 MLATokenToKVPool，buffer shape 是
  (size + page_size, 1, kv_cache_dim)，也就是 3D。
  page-major MHA pool 主要作用在 MHA-shaped pool，不会把这个普通 MLA pool
  自动变成 [num_pages, page_size, head_num, head_dim] 这种 4D MHA view。
```

还有一个容易混淆的环境变量：

```bash
SGLANG_USE_HND_KVCACHE=1
```

它会让普通 `MHATokenToKVPool` 使用 HND 4D layout：

```text
(num_pages, head_num, page_size, head_dim)
```

注意这个 shape 的 dim-1 是 `head_num`，不是 `page_size`。它不是 `_extract_kv_strides` 里处理的 `[num_pages, page_size, head_num, head_dim]` page-major 4D 格式，主要给 HND/paged backend 使用，不能把它和 Triton decode 里的 page-major 4D 分支混为一谈。

结论：

```text
1. 这两个模型在 CUDA/Hopper + attention_backend=triton + 未设置 page_size 时，
   page_size = 1，不会触发 decode_attention.py 的 PAGE_SIZE > 1 分支。

2. 要让这两个模型触发 Triton decode PAGE_SIZE > 1，最明确的命令是：
   --attention-backend triton --page-size 64

3. 4D KV buffer 不是 page_size > 1 自动带来的。
   Llama 这类 MHA 模型需要 --enable-page-major-kv-layout 或 --enable-unified-memory
   才会引入 _extract_kv_strides 支持的 4D page-major KV view。

4. DeepSeek-V2 Lite 的普通 MLA KV pool 仍是 3D；
   它不会因为模型本身是 DeepSeek 或 MLA 就自动变成 4D page-major MHA buffer。
```

## 为什么 cherry-pick 后用 `self.override()` 设置 deterministic sampling

### 结论

这次修改不是要改变来源提交的最终语义。来源提交希望在：

```bash
SGLANG_FORCE_DETERMINISTIC_SAMPLING=1
```

时强制得到：

```python
sampling_backend = "pytorch"
```

之所以没有原样保留：

```python
self.sampling_backend = "pytorch"
```

是因为来源分支和 cherry-pick 目标分支的 `ServerArgs` 已经属于两套不同的配置解析架构：

```text
来源分支 0065170cb:
  各个 handler 直接修改 self.xxx

目标分支:
  handler 先把配置声明存入 self._resolved_overrides
  __post_init__ 末尾再统一 materialize 到 self.xxx
```

在目标分支里只做普通赋值，值会暂时变成 `pytorch`，但随后可能被最终的 `materialize_declarations()` 覆盖回之前声明的 `flashinfer`。`self.override()` 会把这次强制值也追加到有序声明列表里，因此最终物化时仍然是 `pytorch`。

### 来源分支为什么可以直接赋值

来源分支的 `_handle_sampling_backend()` 是旧式命令式写法：

```python
def _handle_sampling_backend(self):
    if self.sampling_backend is None:
        self.sampling_backend = (
            "flashinfer" if is_flashinfer_available() else "pytorch"
        )
```

之后 `_handle_deterministic_inference()` 再执行：

```python
if (
    not self.enable_deterministic_inference
    and os.getenv("SGLANG_FORCE_DETERMINISTIC_SAMPLING", "") == "1"
):
    self.sampling_backend = "pytorch"
```

两次都是直接赋值，且 deterministic handler 后执行，所以后一次赋值自然获胜：

```text
self.sampling_backend = "flashinfer"
self.sampling_backend = "pytorch"
最终结果                 = "pytorch"
```

这个分支没有在最后重放另一份配置声明，因此普通赋值不会再被覆盖。

### 目标分支的执行顺序

cherry-pick 的目标分支已经把 sampling 默认值迁移到了 `python/sglang/srt/arg_groups/overrides.py`。

当前 `ServerArgs.__post_init__` 的相关顺序是：

```text
1. self._resolved_overrides = []
2. self._handle_sampling_backend()
3. self._handle_deterministic_inference()
4. 继续执行其他配置 handler
5. materialize_declarations(self)
```

其中 `_handle_sampling_backend()` 不再直接修改 `self.sampling_backend`，而是调用：

```python
run_post_process_pass(self, _sampling_backend_default)
```

默认值 pass 返回一个声明：

```python
@register_post_process
def _sampling_backend_default(view):
    if view.sampling_backend is None:
        return {
            "sampling_backend": (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )
        }
    return {}
```

`run_post_process_pass()` 在初始化阶段不会立即把这个值写入普通字段，而是把它存入：

```python
self._resolved_overrides
```

假设用户没有显式指定 sampling backend，并且当前环境可以使用 FlashInfer，此时状态近似为：

```text
self.sampling_backend:
  None

self._resolved_overrides:
  [("_sampling_backend_default", {"sampling_backend": "flashinfer"})]
```

如果此时仍使用来源分支的普通赋值：

```python
self.sampling_backend = "pytorch"
```

中间状态会变成：

```text
self.sampling_backend:
  "pytorch"

self._resolved_overrides:
  [("_sampling_backend_default", {"sampling_backend": "flashinfer"})]
```

但 `__post_init__` 末尾会执行：

```python
materialize_declarations(self)
```

它按顺序重放 `_resolved_overrides` 中的字段，因此又会执行等价于：

```python
self.sampling_backend = "flashinfer"
```

最终结果反而错误地变成：

```text
flashinfer
```

如果机器上没有 FlashInfer，默认声明本身就是 `pytorch`，这个错误会被偶然掩盖；但环境变量的强制语义不能依赖这种偶然情况。

### `self.override()` 做了什么

当前 `ServerArgs.override(source, **fields)` 主要完成四件事。

第一，它通过 `resolvable_fields(type(self))` 区分声明式字段和普通运行期字段：

```python
whitelist = resolvable_fields(type(self))
declared = {k: v for k, v in fields.items() if k in whitelist}
rest = {k: v for k, v in fields.items() if k not in whitelist}
```

`sampling_backend` 的参数元数据明确设置了：

```python
Arg(..., resolvable=True)
```

所以它属于 `declared`。

第二，它把声明式字段连同来源标签追加到 `_resolved_overrides`：

```python
stash.append((source, dict(declared)))
```

本次调用会形成类似下面的有序声明：

```text
[
  ("_sampling_backend_default",
   {"sampling_backend": "flashinfer"}),

  ("_handle_deterministic_inference:SGLANG_FORCE_DETERMINISTIC_SAMPLING",
   {"sampling_backend": "pytorch"}),
]
```

声明列表遵循 last-writer-wins，也就是同一字段后面的声明覆盖前面的声明。最终物化顺序因此是：

```text
先写 sampling_backend = "flashinfer"
再写 sampling_backend = "pytorch"
最终 sampling_backend = "pytorch"
```

第三，`override()` 也会立即修改当前对象：

```python
object.__setattr__(self, "_in_override", True)
try:
    self.sampling_backend = "pytorch"
finally:
    object.__setattr__(self, "_in_override", False)
```

这样后续直接读取 `self.sampling_backend` 的代码立刻就能看到新值；通过 `resolved_view(self)` 读取的代码也会从声明 overlay 中看到最后追加的 `pytorch`。

第四，它记录配置变更来源。这里的两个相邻字符串：

```python
"_handle_deterministic_inference:"
"SGLANG_FORCE_DETERMINISTIC_SAMPLING"
```

会由 Python 自动拼成一个字符串：

```text
_handle_deterministic_inference:SGLANG_FORCE_DETERMINISTIC_SAMPLING
```

它只是 provenance/source 标签，便于检查某个值由谁覆盖，不负责读取环境变量，也不参与 backend 选择逻辑。

对于不在 resolvable 白名单中的字段，`override()` 不会放入 `_resolved_overrides`，而是记录到 `_runtime_mutations`；无论属于哪一类，它都会立即写入字段。

### 为什么不能只依赖普通赋值

三种方式的关键区别如下：

| 方式 | 立即修改 `self` | 加入 `_resolved_overrides` | 最终 materialize 后仍保留 | 来源追踪 |
|---|---:|---:|---:|---:|
| `self.sampling_backend = "pytorch"` | 是 | 否 | 不一定 | 否 |
| `self.override(..., sampling_backend="pytorch")` | 是 | 是 | 是 | 是 |
| `run_post_process_pass(...)` | 初始化时否 | 是 | 是 | 是 |

因此这里使用 `override()` 的主要目的不是绕过 Python 赋值限制，而是把环境变量产生的强制值加入目标分支的配置声明顺序，避免最终物化覆盖它。

此外，配置完成后如果开启严格 mutation 检查，普通的 `self.xxx = value` 可能触发 `ServerArgs.__setattr__` 的保护；`override()` 会在 `_in_override` 保护区内执行，是允许且可审计的运行期修改入口。不过本次冲突最直接的问题发生在初始化末尾的 materialize 覆盖，而不是严格检查本身。

### `override()` 是不是最纯粹的声明式写法

严格按当前架构设计，最纯粹的方案是新增一个 post-process pass，例如：

```python
@register_post_process
def _force_deterministic_sampling_env(view):
    if (
        not view.enable_deterministic_inference
        and os.getenv("SGLANG_FORCE_DETERMINISTIC_SAMPLING", "") == "1"
    ):
        return {"sampling_backend": "pytorch"}
    return {}
```

然后在原来的 handler 顺序位置执行：

```python
run_post_process_pass(self, _force_deterministic_sampling_env)
```

这种方式在 `__post_init__` 期间只追加声明，不立即修改原始字段，并且会走 `validate_declarations()`；它比在初始化过程中调用定位为运行期 mutation API 的 `override()` 更符合声明式 pipeline 的风格。

本次冲突解决选择 `self.override()`，是因为它是现有 API 中能以最小改动同时满足以下条件的方式：

```text
1. 保留来源提交的强制 pytorch 语义；
2. 不回退目标分支已经完成的 resolution pipeline 重构；
3. 让环境变量声明晚于 sampling 默认声明，保证 last-writer-wins；
4. 保留来源追踪和后续重新物化的一致性；
5. 不额外修改 overrides.py、注册表和更多测试边界。
```

所以更准确地说：

```text
self.override(...) 是行为正确、改动范围最小的冲突适配方案；
新增独立 post-process pass 是后续若要继续重构时更架构原生的方案。
```

### 最终执行效果

当：

```bash
SGLANG_FORCE_DETERMINISTIC_SAMPLING=1
```

并且：

```text
enable_deterministic_inference = False
```

时，最终声明顺序是：

```text
sampling 默认值声明       -> flashinfer（若可用）
环境变量强制声明          -> pytorch
materialize last writer   -> pytorch
```

结论：

```text
1. 来源分支的直接赋值在旧的命令式 ServerArgs 中是正确的。
2. 目标分支已经改为“先收集声明、最后统一物化”。
3. 直接赋值不进入声明列表，可能在末尾被默认声明覆盖。
4. override 会立即赋值，并把 resolvable 字段追加到声明列表。
5. sampling_backend 是 resolvable 字段，所以后追加的 pytorch 最终获胜。
6. source 字符串只是变更来源标签。
7. 若继续做架构整理，独立 post-process pass 会比 init 中调用 override 更纯粹。
```
