# Kimi-K3 A3 均匀裁剪至 256 个专家

`prune_experts_a3.py` 适配 `Eco-Tech/Kimi-K3-w4a8` 动态 W4A8 权重。
每层按 `keep_ids[j] = floor(j * 896 / 256)` 保留 256 个专家，
重编号为 0–255，按相同顺序提取 gate 和校正 bias。
每个专家的 weight、weight_scale、weight_offset、scale_bias 共 12 个张量
按原始字节复制，不反量化或重新量化。

## 环境和路径

在 Linux 宿主机、仓库根目录运行，Python 3.10 或更高版本：

```bash
python -m pip install safetensors numpy regex
```

本工具使用 Linux `fcntl` 文件锁。共享盘路径需在执行主机上实际存在；
源、输出和工作目录必须互不包含，且不能通过符号链接指定。
源目录需包含主索引、配置、量化描述及其引用的 optional 旋转文件。

## 计划扫描

```bash
python tools/kimi_k3/prune_experts_a3.py \
  --src /mnt/share/c30087248/weights/Eco-Tech/Kimi-K3-w4a8 \
  --dst /mnt/share/c30087248/weights/CuiZ2007/Kimi-K3-w4a8-experts256.partial \
  --work /mnt/share/c30087248/pruning-runs/kimi-k3-a3-w4a8-ep256 \
  --keep-experts 256 --mode plan
```

计划写入工作目录的 `manifests/plan.json` 和 `manifests/expert_map.json`。
源分片读取文件头；辅助文件及 optional 资产读取内容计算 SHA256。
扫描不生成输出权重，但会占用 CPU 和共享盘读取带宽。

## 生成及恢复

核对计划与磁盘空间后，将同一命令的模式改为 `run`：

```bash
python tools/kimi_k3/prune_experts_a3.py \
  --src /mnt/share/c30087248/weights/Eco-Tech/Kimi-K3-w4a8 \
  --dst /mnt/share/c30087248/weights/CuiZ2007/Kimi-K3-w4a8-experts256.partial \
  --work /mnt/share/c30087248/pruning-runs/kimi-k3-a3-w4a8-ep256 \
  --keep-experts 256 --mode run --workers 1
```

可用 `--workers 1..4` 调整分片复制并发。先用 1，避免给共享盘增加过多负载。
写入采用 8 MiB 数据块；计划和元数据仍会占用与张量数量相关的系统内存。
预计输出有效数据 473,582,169,856 字节，另需辅助文件及元数据空间。

中断后保持相同参数、源文件和工作目录，再执行 run 可以恢复。
恢复会核对计划摘要，并回读已完成分片验证其张量哈希。
源主分片身份依据文件大小、修改时间和文件头哈希；源有效数据发生修改后，
应使用新的工作目录重新生成，不能把保留时间戳的源数据修改当作安全恢复场景。

`PRUNING_INCOMPLETE` 表示结果尚未完成。只有任务状态为 completed、
`reports/validation.json` 通过且存在 `PRUNING_COMPLETE.json` 时，才可发布输出。
确认正式目录不存在后，将 `.partial` 目录重命名为
`Kimi-K3-w4a8-experts256`，并在发布记录中保存原路径与正式路径的对应关系。
工具不会自动发布，也不会启动推理服务。

## 校验和输出记录

| 文件 | 用途 |
| --- | --- |
| 工作目录 `status.json` | 阶段、已完成分片、字节数和错误 |
| `manifests/plan.json` | 源文件身份、选取清单、复制计划及摘要 |
| `manifests/expert_map.json` | 新旧专家编号对应关系 |
| `manifests/shards/*.json` | 分片回执、每张量 SHA256 和计划摘要 |
| `reports/validation.json` | 文件级校验结果及未验证项目 |
| 输出 `expert_pruning_manifest.json` | 源路径、选取规则及验证摘要 |
| 输出 `PRUNING_COMPLETE.json` | 完成标记及结果 |

写入后回读全部保留张量，与复制时源字节计算的 SHA256 对比，
并使用 safetensors 独立读取器检查分片。辅助文件按哈希验证，
配置只修改 `text_config.num_experts`；量化描述同步删除并重命名专家条目。
`optional/quarot.safetensors` 原样复制，`group_size=0` 和全局量化类型保持不变。

## 测试

```bash
python -m unittest discover \
  -s tests/ut/tools/kimi_k3 -p 'test_prune_experts_a3.py' -v
```

覆盖均匀映射、专家移除与重编号、非连续路由行、BF16 bias、A3 参数形状、
分片复制与损坏检测、并发复制、源变化、辅助文件与量化描述重映射。
文件级校验不执行专家矩阵乘、CPU 数值路由或 NPU 推理；
完整生产权重生成与 A3 服务验证需另行执行。

详细方案见 [A3 裁剪方案](../../Kimi-K3-W4A8-E256-A3-Pruning.zh-CN.md)。

已在 5.4 Ubuntu 服务器通过 14 项测试及真实源权重计划扫描：
286,737 个输出张量、345 个输出分片、473,582,169,856 字节有效数据。

## 并发保护与完成标记

输出目录旁的 `<dst>.lock` 使用 Linux flock，绑定规范化的输出路径。
即使两个任务的工作目录不同，也只能有一个任务操作同一输出；计划扫描同样持锁。
锁文件保留 inode，进程结束即释放锁，锁文件存在本身不代表任务仍在运行。
不要在任务运行期间删除或替换锁文件。工作目录中的 `job.lock` 继续保护任务记录。

完成时先原子写入 `PRUNING_COMPLETE.json` 并同步目录，再移除未完成标记。
完成标记写入失败时保留未完成标记；若恢复时两种标记同时存在，
先核对计划归属、清除旧完成标记，再按回执重验分片和最终结果。
新增测试覆盖跨进程锁、锁文件软链接、完成写入失败和双标记恢复。
