# MGM-Video ASA 质量探针用法

把 BLADE 风格的二值块稀疏注意力（ASA）套到 MMDiT denoise 的 video↔video 注意力上，
**仅用于肉眼验证视频质量是否受损**——不跳块、无墙钟加速。

详见设计文档：`docs/superpowers/specs/2026-05-30-mgm-video-asa-quality-probe-design.md`。

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `VLLM_MGM_ASA_ENABLE` | `0` | 总开关。`1` 开启 |
| `VLLM_MGM_ASA_TAU` | `0.95` | 累积阈值（越大保留越多、越接近 dense） |
| `VLLM_MGM_ASA_BLOCK` | `128` | 块大小（建议 64 / 128） |
| `VLLM_MGM_ASA_LAYERS` | `all` | 生效层范围，如 `8-41` |
| `VLLM_MGM_ASA_STEPS` | `all` | 生效 step 范围，如 `2-7` |
| `VLLM_MGM_ASA_PER_HEAD` | `1` | 1=逐头掩码；0=保留位（CP 下每 fa 调用本就单头） |
| `VLLM_MGM_ASA_LOG` | `1` | 1=打印每次 layer/step/block_sparsity |

范围格式：`all` 或 `lo-hi`（闭区间，0-based）或单个索引 `n`。

## 跑法

```bash
# 1) dense 基线（同 seed）
VLLM_MGM_ASA_ENABLE=0 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm \
  --prompt "A cat playing piano in a cozy room" \
  --seed 42 --output asa_dense.mp4

# 2) ASA 开启
VLLM_MGM_ASA_ENABLE=1 VLLM_MGM_ASA_TAU=0.95 VLLM_MGM_ASA_BLOCK=128 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm \
  --prompt "A cat playing piano in a cozy room" \
  --seed 42 --output asa_tau095.mp4

# 3) sweep 示例：只对深层、后段 step 稀疏
VLLM_MGM_ASA_ENABLE=1 VLLM_MGM_ASA_LAYERS=8-41 VLLM_MGM_ASA_STEPS=2-7 \
  python examples/offline_inference/text_to_video/mgm_video_t2v.py \
  --model /path/to/mgm_video_11b_vllm --seed 42 --output asa_deep.mp4
```

逐帧/并排比较 `asa_dense.mp4` 与 ASA 输出，结合日志里的 `block_sparsity` 判断
"质量损失 vs 稀疏率"。判定逻辑见设计文档 §5.1。

## 已知限制（诚实标注）

- **无加速**：dense FA + mask 不跳块，总时间只增不减。
- **无 Gilbert 重排**（raster 分块）：这是"质量下界探针"。raster 质量 OK → Gilbert 只会更好；
  raster 崩 → 需补 Gilbert 再判，不能直接下"ASA 不适配"。
- 3.3GB 量级的 `[S,S]` 掩码是 dense-mask 路线的固有显存代价（逐 head 串行、用完即释放）。
