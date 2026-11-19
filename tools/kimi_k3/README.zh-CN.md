# Kimi-K3 A5：均匀裁剪至 256 个专家

来源：`test/kimi-k3-pruned-experts-accuracy` 分支的裁剪方案、脚本和单元测试。
本目录仅包含裁剪工具；不包含精度评测和 nightly 注册。

## 权重裁剪及校验

### 本次使用的原始权重版本

裁剪输入为 **Kimi-K3 W4A8 MXFP QuaRot 权重版本**，本地标识为
`Kimi-K3-w4a8-mxfp-QuaRot`，保留完整 93 层、每个 MoE 层 896 个路由专家。
原始权重目录为 `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot`；
裁剪发布目录为 `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot-slim_ep_256xps`。

本次实验使用原始权重的复制副本，也可以直接使用原始权重目录进行裁剪。
裁剪清单记录的源文件身份校验值如下：

| 源文件 | SHA256 |
| --- | --- |
| `config.json` | `07677f14bc468c16a788fc6517f7fa715701fa1c3ff74b3f7ef3449a5bc8c735` |
| `model.safetensors.index.json` | `b21a53e0b92c0bb6474be2da74ceb9261ba888f8880a8b8f84fb8b8d6ee6e4e1` |

现有记录未提供该本地权重对应的上游仓库 revision 或发布 tag。
以上目录标识和校验值用于定位本次实验输入；校验值覆盖配置、索引文件，
并非全部源权重数据的哈希。第 8 节的 vLLM、vllm-ascend 版本属于运行环境版本，
不属于原始权重版本。

独立裁剪依赖安装命令为 `python -m pip install safetensors numpy regex`。

输入原始量化权重；每个 MoE 层从 896 个路由专家中均匀选取 256 个：
`keep_ids[j] = floor(j * 896 / 256)`，其中 `j=0..255`。
92 个 MoE 层均采用此规则；专家编号重映射为 0～255，Router 权重和校正 bias
按同一清单切片。量化权重及 scale 保留原始字节，不反量化或重新量化。
完整保留 93 层、每 token 激活 16 个专家和两个共享专家。
这是均匀容量裁剪，不是基于重要性校准的选择，也不保证精度保持不变。

```bash
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode plan
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode run --workers 4
```

命令中的路径需要替换。保留专家映射、复制计划和分片校验记录，确认全部完成后再发布
临时输出为最终权重。个人服务器 Docker 发布及挂载脚本未迁入工具。
历史裁剪报告记录：358 个输出分片、144683 个张量，保留张量通过逐字节校验。
复制计划摘要为：
`6ed94726e172e3b67c344901a70383451308446f6f346cd64213bd0094533f1e`。
这不是完整权重哈希；整理后的工具尚未重新处理全量权重。


## 文件与单元测试

- `prune_experts.py`：裁剪计划生成、原始字节复制和分片校验。
- `../../tests/ut/tools/kimi_k3/test_prune_experts.py`：裁剪工具单元测试。

在仓库根目录执行：

```bash
python -m unittest discover -s tests/ut/tools/kimi_k3 -p 'test_prune_experts.py' -v
```
