# Test report consolidated

See the complete [English document](README.md) or [中文版](README.zh-CN.md),
section 8 for three-round results and conclusions.

## A5 精度测试环境记录

以下软件版本来自此前三轮 GSM8K 精度测试所用旧开发环境的保存快照，
并非当前分支或正式 `v0.30.0-a5` 镜像的运行验证结果。
环境快照来源为 `outputs/component-repair-evidence/reports/main-readiness/environment.json`
（本地看护工作区证据文件，不随本报告提交）。数据集路径采用当前整理后的目录。

### 容器外：宿主机与共享资源

| 项目 | 配置 |
| --- | --- |
| 服务器 | `141.61.41.213` |
| 硬件 | 单节点 A5，8 张 NPU |
| 架构 | ARM64 / aarch64 |
| Ascend 驱动 | `25.6.rc3` |
| 裁剪权重 | `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot-slim_ep_256xps` |
| 当前 GSM8K 数据目录 | `/mnt/share/c30087248/datasets/gsm8k` |

环境快照确认容器内可访问 `/dev/davinci0` 至 `/dev/davinci7`，以及
`/usr/local/Ascend/driver`。容器运行还需要可读的权重和数据集挂载。
快照中的历史数据挂载路径为 `/mnt/share/c30087248/gsm8k`。
宿主机操作系统、Docker 版本和完整容器创建参数尚未从本次证据核实。

### 已核实的基础镜像

历史测试容器基于以下 ARM64 / aarch64 nightly 镜像修改而来：

```text
quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5
```

服务器上的镜像 ID 为：

```text
sha256:4e34d514c73d903e1860a9034275dd855c14511bf35a01f75601ae25980610e6
```

镜像归档文件位于：

```text
/mnt/share/c30087248/vllm-ascend_images/vllm-ascend-nightly-releases-v0.30.0-a5-arm64.oci.tar
```

经服务器 Docker 镜像层比对，nightly 镜像的 18 个基础层完整构成本地快照
`c30087248/kimi-k3-slim:before-components-20261031`（23 层）的前缀。
该快照 ID 为 `sha256:a8fc248b64bf4031d508b764bbb4bd7808bd0687980d4be9341157f2dbbc0c2e`。
这确认了镜像继承关系；容器内后续安装依赖与修改源码后的实际软件版本见下表，
不能将其视为未经修改的 nightly 镜像版本清单。
历史测试未换用另行下载的正式 `quay.io/ascend/vllm-ascend:v0.30.0-a5` 镜像。

### 容器内：历史实际软件版本

| 组件 | 版本 |
| --- | --- |
| Python | `3.12.13` |
| CANN Toolkit | `9.2.0` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post7` |
| vLLM | `0.30.0+empty` |
| vllm-ascend | `0.30.0rc2.dev304+geac9c0cf3.d20261031` |
| Transformers | `5.14.1` |
| Triton | `3.6.0` |
| triton-ascend | `3.6.0.post2` |
| safetensors | `0.8.0` |

PyTorch 的 `+cpu` 是发行版本标签，NPU 支持由 `torch_npu` 提供。
vLLM 与 vllm-ascend 分别以可编辑方式安装自 `/vllm-workspace/vllm`
和 `/vllm-workspace/vllm-ascend`。

### 单文件 YAML 草案的环境变量

以下变量记录新分支 `test/kimi-k3-ep256-yaml` 的
`Kimi-K3-W4A8-EP256-A5-native.yaml` 草案配置，不能视为历史三轮运行时变量的完整快照。

```bash
unset HCCL_OP_EXPANSION_MODE

export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
```

### 适用范围与验证状态

历史结果基于旧开发容器，未验证后来下载的正式 `v0.30.0-a5` 镜像，
也不能作为当前分支 nightly 运行验证。新 YAML 使用 AISBench 原生入口，
目标评测环境需要安装 AISBench 及项目测试依赖；所引用历史快照中 AISBench 未安装。
新 YAML 尚未在目标环境启动验证，尤其需要确认 Kimi-K3 支持和
`--default-chat-template-kwargs` 参数可用性，并标定原生评分的准确率基线与容差。
