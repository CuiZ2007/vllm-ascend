# Kimi-K3.1-Pro W4A8 E32 A3 裁剪说明

## 仓库新增实现

本分支新增 `tools/kimi_k31/prune_experts_a3.py`，使用方法见
[裁剪脚本说明](tools/kimi_k31/README.md)。该实现依据下方方案重新编写，
已进行合成权重测试，尚未在完整生产权重上执行；下方历史结果属于原服务器脚本。

> 目标：生成独立的 32 路由专家权重，同步裁剪 MoE gating，保留原权重，保持 attention 和其他保留权重的数值及数据类型不变。
>
> 验证结论：文件、配置、量化描述、保留张量哈希及 CPU 路由检查通过；NPU 整网加载、推理启动和精度尚未验证。

## 1. 本次结果

| 项目 | 原权重 | 裁剪后 |
|---|---|---|
| 目录 | `/mnt/weight/Kimi-K3.1-Pro-W4A8` | `/mnt/weight/Kimi-K3.1-Pro-W4A8-E32` |
| 每个 MoE 层的路由专家数 | 896 | 32 |
| 保留的原始专家编号 | 0–895 | 0–31 |
| MoE 层数 | 92 | 92 |
| 每 token 选择专家数 | 16 | 16，未修改 |
| 每个 MoE 层的共享专家数 | 2 | 2，未修改 |
| 专家分组参数 | `num_expert_group=1`、`topk_group=1` | 未修改 |
| 张量数 | 993,464 | 39,608 |
| safetensors 分片数 | 345 | 339 |
| 张量有效数据体积 | 1,466,473,049,856 字节，约 1,365.76 GiB | 126,210,938,112 字节，约 117.54 GiB |
| attention 张量有效数据 | 53,338,992,384 字节，约 49.68 GiB | 全量保留，字节一致 |
| 配置变更 | `text_config.num_experts=896` | `text_config.num_experts=32` |

删除了 **953,856 个被裁专家所属的张量**；另有 **184 个路由张量被裁剪形状**，这 184 个张量仍计入最终 39,608 个张量。

按张量有效数据计算，体积减少约 **91.39%**。这里的 GiB 按 `字节数 / 1024³` 计算，不包含分片头、索引、tokenizer 和校验报告等额外文件。

本次采用**按编号保留前 32 个专家**的方法，没有根据校准数据、路由频率或专家重要性排序，也没有重新训练。

## 2. 具体裁剪了什么

### 2.1 路由专家：保留编号 0–31，删除编号 32–895

路由专家的张量名具有以下结构：

```text
language_model.model.layers.{layer}.block_sparse_moe.experts.{expert_id}.{projection}.{parameter}
```

本次检查到 MoE 位于文本模型第 1–92 层，层编号从 0 开始；第 0 层为原来的 dense MLP，保持不变。

每个路由专家有三组投影：

- `w1`
- `w2`
- `w3`

每组投影在该量化权重中都有四个张量：

- `weight`
- `weight_scale`
- `weight_offset`
- `scale_bias`

因此每个专家对应 **3 × 4 = 12 个张量**。裁掉一个专家时，这 12 个张量一起移除；保留一个专家时，这 12 个张量一起按原字节复制。

核心判断等价于：

```python
if is_routed_expert_tensor(name):
    if expert_id >= 32:
        continue  # 不写入新权重，也不保留其索引和量化描述
    copy_tensor_bytes_exactly()
```

删除张量数可以直接核对：

```text
每层删除专家数 = 896 - 32 = 864
删除张量数     = 92 × 864 × 12 = 953,856
最终张量数     = 993,464 - 953,856 = 39,608
```

由于保留的是连续编号 0–31，**不需要重命名或重新编号专家**。新专家 0 仍然对应原专家 0，新专家 31 仍然对应原专家 31。

### 2.2 MoE gating：按同一专家集合裁剪行

本次实际检查到每个 MoE 层有两个路由张量：

```text
language_model.model.layers.{layer}.block_sparse_moe.gate.weight
language_model.model.layers.{layer}.block_sparse_moe.gate.e_score_correction_bias
```

裁剪规则如下：

| 张量 | 原形状 / 类型 | 新形状 / 类型 | 处理方式 |
|---|---|---|---|
| `gate.weight` | `[896, 7168]`，BF16 | `[32, 7168]`，BF16 | 保留第 0–31 行 |
| `gate.e_score_correction_bias` | `[896]`，FP32 | `[32]`，FP32 | 保留第 0–31 项 |

数值关系为：

```python
new_gate_weight = original_gate_weight[:32, :]
new_correction_bias = original_correction_bias[:32]
```

这里的代码用于说明数值关系。实际生成时按 safetensors 中的原始字节区间复制，没有把 BF16 转成 FP32 再保存。

两个张量的第 0 维都与专家编号一致，因此：

```text
新专家 i
  ↔ 原专家 i
  ↔ 原 gate.weight 第 i 行
  ↔ 原 e_score_correction_bias 第 i 项
其中 i = 0, 1, ..., 31
```

需要同步裁剪 correction bias，是因为它参与路由选择；只裁 gate 矩阵而保留原来的 896 项 bias，会造成专家维度不一致。

92 个 MoE 层各裁剪 2 个路由张量，共：

```text
92 × 2 = 184 个路由张量
```

### 2.3 为什么没有误裁 attention 内部的 gating

匹配规则限定为：

```regex
^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$
```

因此只有 **MoE 路由器**的上述两个张量会裁剪。attention 内部即使有名字含 gate 的参数，也不会命中这个规则。

同样，路由专家的删除规则限定为：

```regex
^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$
```

它不会匹配 `shared_experts`，也不会把名称中包含 `routed_expert` 的公共投影误当成独立专家。

### 2.4 原样保留的部分

以下张量均保留原 shape、dtype 和有效数据字节：

- 全部 attention，包括 KDA、DSA 及相关投影、归一化、卷积、内部 gate 和量化参数。
- 两个共享专家对应的全部权重与量化参数。
- `routed_expert_down_proj`、`routed_expert_up_proj`、`routed_expert_norm` 等公共 latent MoE 变换。
- 第 0 层 dense MLP。
- embedding、lm_head、其他 norm。
- vision tower 和多模态 projector。
- 保留的专家 0–31 对应的所有 W4A8 量化数据及辅助参数。

tokenizer、自定义配置代码、预处理文件及其他辅助文件也复制到新目录；复制的辅助文件另做 SHA256 核对。

## 3. 实际执行流程

### 步骤一：连接服务器并检查资源

通过 MindIE remote-dev 的 SSH 能力连接服务器，读取权重目录和环境信息。最初连接超时，用户连接 VPN 后恢复访问。

本次检查包括：

1. 源目录是否存在。
2. config、权重索引、量化描述文件是否齐全。
3. 存储空间、系统内存以及 NPU 设备信息。
4. 服务器上已有的推理容器和可用于独立验证的 Python 环境。

探测时看到 16 个 Ascend 计算设备，每个报告约 64 GiB HBM；模型所在存储剩余约 10.6 TiB，系统可用内存约 2.9 TiB。

权重改写在宿主机运行；独立验证使用已有容器。该验证容器中的 `/mnt/weight` 是只读挂载，验证脚本不会通过容器改写权重。

### 步骤二：读取配置、索引与量化描述

读取三个主要文件：

```text
config.json
quant_model_weights.safetensors.index.json
quant_model_description.json
```

确认的关键配置是：

```json
{
  "num_hidden_layers": 93,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1,
  "num_experts": 896,
  "num_experts_per_token": 16,
  "num_shared_experts": 2,
  "num_expert_group": 1,
  "topk_group": 1,
  "hidden_size": 7168
}
```

上述字段位于 `text_config` 内，示例只列出相关字段。

随后确认：

- 每个 MoE 层都有 896 个专家。
- 每个专家恰好有前述 12 个张量。
- 每层两个路由张量的 shape 和 dtype 符合预期。
- `num_experts_per_token=16` 不超过目标专家数 32。
- 原模型已经是单组专家，因此本次不需要修改专家分组参数。

### 步骤三：扫描全部 safetensors 文件头，建立裁剪计划

生成脚本只依赖 Python 标准库，读取每个 safetensors 文件头中的：

- 张量名。
- dtype。
- shape。
- `data_offsets`，即张量在数据区中的起止字节位置。

扫描阶段核对了：

1. 张量名没有重复。
2. 索引中的分片名与张量实际所在分片一致。
3. 数据区间连续，没有空洞或重叠。
4. 文件头长度与数据区长度之和等于实际文件大小。
5. 扫描得到的张量集合与原索引完全一致。
6. 所有原张量有效数据长度之和等于原索引的 `metadata.total_size`。

这一步覆盖了原来的 **993,464 个张量**。未保留的专家只需要检查头部和索引，不需要将其全部数据读入内存。

计划中的每个保留张量记录：

```text
key
source_file
source_offset
length
source_descriptor
output_descriptor
action = copy_exact 或 crop_router_rows
```

### 步骤四：创建独立的临时输出目录

本次先写入：

```text
/mnt/weight/Kimi-K3.1-Pro-W4A8-E32.partial
```

脚本为目标目录建立独占生成锁；若最终目录、临时目录或生成锁已经存在，会拒绝相应的重复生成操作，避免覆盖已有结果。

新文件不与原权重建立硬链接或软链接。源权重以只读方式打开；所有权重数据写入新的文件。

### 步骤五：按字节复制保留张量，并同步裁剪路由

对普通保留张量：

```text
起点 = 源文件数据区起点 + 原 data_offsets[0]
长度 = 原 data_offsets[1] - 原 data_offsets[0]
复制该完整字节区间
```

对两个路由张量：

```text
每个专家对应的字节数 = 原张量有效数据长度 / 896
新有效数据长度       = 每个专家对应的字节数 × 32
从原张量起点复制这一段前缀数据
将新 shape 的第 0 维改为 32
```

例如，单层 gate 矩阵由：

```text
896 × 7168 × 2 字节 = 12,845,056 字节
```

变成：

```text
32 × 7168 × 2 字节 = 458,752 字节
```

每次最多读取约 **16 MiB** 的数据块，不会把 1.4 TB 量级的原权重一次载入内存，也不需要启动 NPU。

本次没有反量化、重新量化或重新计算 scale/offset。保留的 W4A8 数据保持原有存储表示。

### 步骤六：重写新分片及配套元数据

#### 6.1 safetensors 分片

每个仍有保留张量的源分片，对应生成一个新的分片：

1. 按新保留集合重新安排连续的数据区。
2. 重新计算每个张量的 `data_offsets`。
3. 写入新的 JSON 文件头。
4. 若原分片存在 `__metadata__`，复制该元信息。
5. 写入保留张量的数据。
6. 刷新并同步文件，再将分片临时文件改为正式文件名。

345 个源分片中有 6 个裁剪后不再含任何张量，因此最终有 339 个新分片，命名形式为：

```text
quant_model_weights-00001-of-00339.safetensors
...
quant_model_weights-00339-of-00339.safetensors
```

分片数量接近原模型，是因为其他保留张量分布在许多源分片中；它不代表专家没有被删除。本次没有额外把这些分片重新合并成固定 4 GiB 大小。

#### 6.2 权重索引

重建 `quant_model_weights.safetensors.index.json`：

- 移除被裁专家的所有 key。
- 将保留 key 映射到新分片。
- 将 `metadata.total_size` 更新为 **126,210,938,112** 字节。

#### 6.3 量化描述

重建 `quant_model_description.json`：

- 移除被裁专家的所有张量条目。
- 保留张量的描述值与原文件一致。
- 保留原来的五类全局条目：

```text
group_size
metadata
model_quant_type
optional
version
```

裁剪后量化描述的张量 key 必须与新权重索引一致；不留下指向已删除专家的条目。

#### 6.4 config.json

仅修改一个配置项：

```diff
- "num_experts": 896
+ "num_experts": 32
```

该字段位于 `text_config` 中。脚本递归比较新旧配置，要求差异恰好为：

```text
text_config.num_experts: 896 -> 32
```

层数、attention 配置、KDA/DSA 层分布、hidden size、每 token 选 16 个专家、共享专家数、激活函数及其他配置均不变。

### 步骤七：执行全量哈希回读

复制时，对每个保留张量从源文件读取的数据计算 SHA256，并记录：

- 原张量 key。
- 源分片及源字节偏移。
- 新分片。
- 复制字节数。
- 处理方式。
- SHA256。

写完后，重新打开全部输出分片：

1. 检查文件头、数据区间和文件长度。
2. 检查张量集合、dtype、shape 和数据长度。
3. 从输出文件回读每个张量，重新计算 SHA256。
4. 与复制时源数据的 SHA256 比较。
5. 同时核对输出分片整体 SHA256。
6. 再次读取 config、索引和量化描述，确认与计划一致。

比较的对象是**张量有效数据**。新分片的文件头、偏移和文件名经过重写，因此不能要求新旧 safetensors 整文件 SHA256 相同。

这里“全部输出张量通过校验”包括两类：

- 普通保留张量：与原张量的完整有效数据一致。
- 裁剪后的路由张量：与原张量对应的前 32 行/项一致。

最终 **39,608 个输出张量**全部通过。

### 步骤八：用独立的 safetensors / PyTorch 环境验证

全部分片写完、生成任务进入哈希回读后，另启动 CPU 验证任务，读取已写完的临时目录。两个验证过程都是读取操作。

本次独立验证使用：

- safetensors **0.8.0**。
- PyTorch **2.10.0+cpu**。
- Transformers AutoConfig，离线读取本地配置。

检查内容：

| 检查项 | 实际结果 |
|---|---|
| 用 `safe_open(..., framework='pt', device='cpu')` 打开所有 339 个分片 | 通过 |
| 枚举全部 39,608 个张量，与新索引核对 | 通过 |
| 每个 MoE 层的专家编号集合均为 0–31 | 通过 |
| 所有保留张量 shape 与原模型一致，路由第 0 维除外 | 通过 |
| 92 个 gate 矩阵与原矩阵前 32 行进行 `torch.equal` 比较 | 通过 |
| 92 个 correction bias 与原向量前 32 项进行 `torch.equal` 比较 | 通过 |
| 路由参数数值有限，未出现 NaN/Inf | 通过 |
| 保留张量的量化描述与原模型一致 | 通过 |
| AutoConfig 读取后得到 32 个专家、top-k 16、共享专家 2 | 通过 |
| KDA/DSA 配置读取后仍与原配置一致 | 通过 |

还对 92 层分别执行了 CPU 路由检查。随机种子为 123，构造 3 个 token 的输入，输入形状为 `[3, 7168]`，计算过程为：

```python
scores = torch.sigmoid(hidden @ gate_weight.float().T)
indices = (scores + correction_bias).topk(16, dim=-1).indices
selected_scores = scores.gather(-1, indices)
routing_weights = selected_scores / selected_scores.sum(-1, keepdim=True)
```

确认：

- 选出的专家索引形状为 `[3, 16]`。
- 所有专家编号都处于 0–31。
- 路由权重都是有限值。
- 每个 token 的路由权重之和约为 1。

该 CPU 检查覆盖路由段，没有执行量化专家矩阵乘、attention 内核或整网前向。

### 步骤九：核对原文件并切换为最终目录

生成前记录源目录所有文件的：

```text
文件大小
mtime_ns
inode
mode
```

生成后检查文件集合及上述属性均未变化；对复制的辅助文件还进行了源文件与输出文件的 SHA256 核对。

本次没有对原模型全部 1.33 TiB 数据额外做一次完整文件哈希扫描。原权重保护依据是：源文件仅以只读模式打开、写入均发生在新目录、源文件属性前后核对，以及保留张量从源读取后的哈希链。

全部生成任务内部校验通过后，在同一父目录内执行：

```python
os.rename(
    "/mnt/weight/Kimi-K3.1-Pro-W4A8-E32.partial",
    "/mnt/weight/Kimi-K3.1-Pro-W4A8-E32"
)
```

独立 CPU 验证也已通过。切换后仍是同一批经过验证的文件，没有重新写入权重。

最终目录为：

```text
/mnt/weight/Kimi-K3.1-Pro-W4A8-E32
```

生成任务报告的写入与全量回读阶段耗时约 **757.4 秒，即 12 分 37 秒**，不含前期连接与结构分析。

## 4. 实际执行命令及复现方式

### 4.1 本次生成使用的命令

生成脚本保存在服务器：

```text
/mnt/weight/.kimi-e32-work/prune_kimi_e32.py
```

本次执行的是：

```bash
python3 -u /mnt/weight/.kimi-e32-work/prune_kimi_e32.py \
  --source /mnt/weight/Kimi-K3.1-Pro-W4A8 \
  --output /mnt/weight/Kimi-K3.1-Pro-W4A8-E32 \
  --keep 32
```

**此命令已经成功执行，当前目标目录已存在。再次执行会拒绝覆盖，不能将它理解为需要补执行的步骤。**

原脚本依赖 Python `assert` 做完整性检查，应使用普通 Python 模式运行，不使用 `-O` / `-OO`，并确保没有设置 `PYTHONOPTIMIZE`。

### 4.2 如需重新生成另一个独立副本

以下是复现示例，本次未额外执行。必须选用一个不存在的输出目录，并确认剩余空间足够：

```bash
SOURCE=/mnt/weight/Kimi-K3.1-Pro-W4A8
OUTPUT=/mnt/weight/Kimi-K3.1-Pro-W4A8-E32-rebuild

# 先扫描、计算裁剪计划；不创建权重副本
env -u PYTHONOPTIMIZE python3 -u \
  /mnt/weight/.kimi-e32-work/prune_kimi_e32.py \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --keep 32 \
  --plan-only

# 计划核对后，写入并执行全量校验
env -u PYTHONOPTIMIZE python3 -u \
  /mnt/weight/.kimi-e32-work/prune_kimi_e32.py \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --keep 32
```

脚本针对本次 Kimi-K3.1 权重的张量命名、专家结构和路由形状编写，并设有相应断言，不应直接当作所有 MoE 模型通用的裁剪工具。

如果运行中断，先检查原任务状态、生成锁和临时目录；不要直接重复提交，也不要在未确认任务结束前删除锁文件或临时结果。

### 4.3 独立 CPU 验证的复现命令

验证脚本位于：

```text
/mnt/weight/.kimi-e32-work/validate_e32.py
```

本次使用的已有容器 ID：

```text
f76474ef693520d319a1e5d5c5fcbe2f494671972e694eafef8d6bc1350cf17a
```

原验证针对已经写完的 `.partial` 目录进行；目前该目录已经切换为最终目录。下面给出使用最终目录重新验证的命令：

```bash
docker exec \
  -e PYTHONOPTIMIZE= \
  -e HF_MODULES_CACHE=/tmp/kimi-e32-validation-hfmodules \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e ASCEND_RT_VISIBLE_DEVICES= \
  -e CUDA_VISIBLE_DEVICES= \
  f76474ef693520d319a1e5d5c5fcbe2f494671972e694eafef8d6bc1350cf17a \
  python /mnt/weight/.kimi-e32-work/validate_e32.py \
    --source /mnt/weight/Kimi-K3.1-Pro-W4A8 \
    --checkpoint /mnt/weight/Kimi-K3.1-Pro-W4A8-E32 \
    --report /tmp/kimi-e32-validation-recheck.json
```

该命令只用于 CPU 验证，不会启动 NPU 推理服务。验证报告写入容器的 `/tmp`，不写只读挂载的权重目录。复现时需要该容器仍存在且保留对应目录挂载。

## 5. 输出目录与校验凭据

最终目录主要包含：

```text
Kimi-K3.1-Pro-W4A8-E32/
├── config.json
├── quant_model_description.json
├── quant_model_weights.safetensors.index.json
├── quant_model_weights-00001-of-00339.safetensors
├── ...
├── quant_model_weights-00339-of-00339.safetensors
├── tokenizer 与原辅助文件
├── pruning_manifest.json
├── pruning_tensor_manifest.jsonl
├── SHA256SUMS.json
└── validation_report.json
```

| 文件 | 用途 |
|---|---|
| `pruning_manifest.json` | 原/新专家数、保留编号、张量数量、字节数、配置差异、源文件属性、校验结论和输出分片 SHA256 |
| `pruning_tensor_manifest.jsonl` | 每个输出张量的源分片、源偏移、复制长度、处理方式及 SHA256 |
| `SHA256SUMS.json` | 输出分片及 config、索引、量化描述、逐张量清单的 SHA256 |
| `validation_report.json` | 全量哈希和独立 CPU 验证的汇总结果，以及未验证项目 |

本地交付文件与本文位于同一 outputs 目录：

- `prune_kimi_e32.py`：生成脚本。
- `validate_e32.py`：独立 CPU 验证脚本。
- `Kimi-K3.1-Pro-W4A8-E32-validation.json`：汇总校验报告。

为追溯执行，远端管理任务编号为：

```text
生成与全量哈希校验：job-20261105T074008Z-51f34c3d
独立 CPU 验证：    job-20261105T074626Z-fc929310
```

## 6. 如何理解“attention 不动”和“单机能跑”

### 6.1 attention 权重不变，不等于整网输出不变

已经证明的是：attention 的参数、量化参数、shape、dtype 和数据字节保持一致。

但删掉专家后，每层可选专家集合从 896 个变成 32 个。原来可能选中的专家 32–895 已经不存在，剩余专家会重新参与 top-k 选择和归一化。因此 MoE 输出可能改变，后续层输入的 hidden states 也可能改变。

即使 attention 权重完全不变，整个裁剪模型的输出也不能据此认定与原模型一致。

### 6.2 本次“单机”判断覆盖的范围

现场资源显示单机有 16 个计算设备，每个约 64 GiB HBM；117.54 GiB 的张量有效数据从单机总容量角度可容纳。

另外检查了现有 Kimi 模型实现：gate 输出维度、correction bias 长度、FusedMoE 专家数及专家加载映射均从配置的 `num_experts` 派生；所检查的构造和映射逻辑没有固定为 896。

这些证据支持**单机容量和专家数量配置的可行性**，但不能替代真实启动。实际部署还涉及每设备内存分布、权重重排、工作区、缓存、通信及对应内核约束。

### 6.3 尚未验证的项目

| 项目 | 当前状态 |
|---|---|
| 原文件保护、保留张量字节一致性 | 已验证 |
| 分片、索引、量化描述和配置一致性 | 已验证 |
| 32 专家的 CPU 路由检查 | 已验证 |
| NPU 上加载完整模型 | 未执行 |
| NPU 量化专家 / attention 内核及整网推理 | 未执行 |
| 单机服务启动与文本生成 | 未执行 |
| 吞吐、时延、最大上下文和并发 | 未测量 |
| 语言、多模态任务精度与原模型对比 | 未评测 |

本次 NPU 验证没有启动，执行记录中的原因是所需设备租约组件不可用。生成与 CPU 校验均不依赖 NPU，因此权重交付已完成，但报告没有将容量检查表述为 NPU 启动成功。

这版权重是按固定专家编号直接裁剪得到的结构精简版本。没有做专家重要性筛选、校准或恢复训练，不能保证精度保持。
