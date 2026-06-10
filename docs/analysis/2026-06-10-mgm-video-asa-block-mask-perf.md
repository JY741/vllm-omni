# MGM-Video ASA `build_asa_block_mask` 性能优化总结

**Date:** 2026-06-10
**模块:** `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py`
**函数:** `build_asa_block_mask`
**硬件:** Ascend NPU
**相关提交:** `299c8e82`, `c8ee4e70`, `1f29ec91`

## 1. 背景

ASA（Adaptive Sparse Attention）在 MGM-Video MMDiT 中按"能量阈值"
对每个 query 块挑选若干最相关的 key 块，构造 block-sparse 注意力掩码。
`build_asa_block_mask` 是这条路径上的核心函数：

```
输入 importance: [B, 1, nq, nk] fp32（块重要度，来自 sample_pool_attn 的 head-mean softmax）
输出 mask:        [B, 1, nq, nk] bool（True = 保留该 KV 块）
```

典型生产 shape：`[1, 1, 451, 451]`，`max_retain_ratio=0.20` → `max_keep ≈ 90`
（profile 用例中实测 `max_keep=315`，下文以此为参考）。

Profiling 显示这一步原本耗时 **~8 ms**，与同层的 `sample_pool_attn`
(~5 ms，含 Q·K + softmax，NPU cube 强项) 相比异常突出。本文记录三轮
优化的根因分析与落地。

## 2. 原始实现（commit 之前的 `dfc7620d`）

```python
def build_asa_block_mask(importance, max_retain_ratio, min_retain_ratio, energy_threshold):
    B, H, nq, nk = importance.shape
    imp_f = importance.float()

    # 1) 全长降序排序
    sorted_imp, indices = torch.sort(imp_f, dim=-1, descending=True)
    # 2) 全长前缀和
    cum = torch.cumsum(sorted_imp, dim=-1)
    total = cum[..., -1:].clamp(min=1e-30)

    # 3) 找首个达到阈值的位置（argmax + any + where 三连）
    over = cum >= energy_threshold * total
    k_indices = torch.argmax(over.int(), dim=-1)
    unsatisfied = ~over.any(dim=-1)
    k_indices = torch.where(unsatisfied, torch.full_like(k_indices, nk), k_indices)
    k_indices = k_indices + 1

    # 4) clamp 到 [min_keep, max_keep]
    min_keep = max(1, int(nk * min_retain_ratio))
    max_keep = max(min_keep, int(nk * max_retain_ratio))
    k_indices = k_indices.clamp(min=min_keep, max=max_keep)

    # 5) 在排序坐标上构造布尔模板，scatter_ 写回原坐标
    pos = torch.arange(nk, device=importance.device).view(1, 1, 1, nk)
    keep_sorted = pos < k_indices.unsqueeze(-1)
    mask = torch.zeros_like(imp_f, dtype=torch.bool)
    mask.scatter_(-1, indices, keep_sorted)
    return mask
```

写法本身在 GPU 上是教科书级别的实现：sort + cumsum + scatter，
每一步都是一行 PyTorch、可读性很高。问题全在 **NPU 后端**。

## 3. 为什么这么写在 NPU 上会跑到 ms 量级

NPU 的 vector / scalar 计算与 cube（GEMM）单元的吞吐差距比 GPU 大得多，
而排序、前缀扫描、散射写这三类算子都**结构性地不能映射到 cube**。

| 算子 | 计算特征 | NPU 后端实现 | 该 shape 下耗时 |
|---|---|---|---|
| `torch.sort(.., descending=True)` | O(nk·log nk)，串行性强 | aclnn `Sort`，merge-sort 风格，长度数百时效率差 | 高 |
| `torch.cumsum(.., dim=-1)` | 严格串行（`y[j]` 依赖 `y[j-1]`） | aclnn `Cumsum`，串行前缀扫描；无法走 cube | **~3 ms / 单次** |
| `torch.scatter_(-1, indices, src)` | 写散射（非连续地址写入） | aclnn `Scatter`，需要逐元素地址解析 + 顺序写 | **本轮原始实现里的最大热点** |

此外，原版叠加了一堆"小但多"的算子：
- `over.int()` 多一次 dtype cast；
- `argmax` + `any` + `where` 是 3 个独立 reduce / elementwise kernel；
- `imp_f = importance.float()` 即使 importance 已是 fp32 也会触发；
- 总计 ~18 个 kernel launch，每次 ~50–100 µs 的 launch overhead 又加 ~1–2 ms。

三大昂贵算子 + kernel launch 风暴 → 8 ms 完全成立。
反观上游的 `sample_pool_attn` 主要是 `matmul + softmax`（cube 强项），
所以 "5 ms vs 8 ms" 的反差并非异常。

## 4. 优化方案三步走

### 第一步：用 `topk(max_keep)` 替换全长 `sort` + 全长 `cumsum`

**关键观察**：最终结果会被 `clamp(max=max_keep)`，其中
`max_keep = int(nk * max_retain_ratio)`（默认 0.20，约为 nk/5）。
也就是说**最多只可能保留前 max_keep 个块**，对剩余 ~4·nk/5 个块
做排序和前缀和是纯浪费。

```python
# 旧：全 nk 排序 + 全 nk cumsum
sorted_imp, indices = torch.sort(imp_f, dim=-1, descending=True)
cum = torch.cumsum(sorted_imp, dim=-1)
total = cum[..., -1:].clamp(min=1e-30)

# 新：topk 拿前 max_keep 个 + 短 cumsum + 单 reduce 拿总和
topk_vals, topk_idx = torch.topk(imp_f, max_keep, dim=-1, sorted=True)
total = imp_f.sum(dim=-1, keepdim=True).clamp(min=1e-30)
cum = torch.cumsum(topk_vals, dim=-1)   # 长度从 nk 降到 max_keep
```

**收益**：sort 长度 nk → topk 长度 max_keep（~5×），cumsum 长度同步缩短。
总能量改用单次 `sum`，不再依赖"全长 cumsum 末尾切片"。

同时把 `argmax + any + where` 三连塌成一个 reduce：

```python
# 旧：3 个 kernel 拼出 k_indices，再 +1，再 clamp
over = cum >= energy_threshold * total
k_indices = torch.argmax(over.int(), dim=-1)
unsatisfied = ~over.any(dim=-1)
k_indices = torch.where(unsatisfied, torch.full_like(k_indices, nk), k_indices)
k_indices = k_indices + 1

# 新：单 reduce，超界时自然落到 max_keep+1，被后续 clamp 拉回
need = energy_threshold * total
k_count = (cum < need).sum(dim=-1, keepdim=True) + 1
k_count = k_count.clamp(min=min_keep, max=max_keep)
```

**等价性证明**：
- 若 cum 在第 i 位首次 ≥ need，则 `cum < need` 在 `[0..i-1]` 为 True、`[i..]` 为 False；`.sum() = i`；`+1 = i+1`，对应"保留 i+1 个块"，与原 `argmax + 1` 一致。
- 若 cum 永远 < need，`.sum() = max_keep`；`+1 = max_keep + 1`，clamp 后回到 max_keep，对应原 `unsatisfied → nk → clamp` 路径。

**第一轮结果**：8 ms → ~3 ms。`sort` 与 `cumsum` 都缩短，但
**scatter 还在**（虽然写入长度也降到了 max_keep）。

> 提交：`299c8e82 perf(mgm_video): replace sort+full-cumsum in ASA block mask with topk`

### 第二步：彻底丢掉 `scatter_`，用 elementwise `imp >= threshold`

第一轮里 `scatter_` 还在，因为我们仍然按"排序坐标 → 原坐标"的路子在
反向回填。但只要观察 mask 的语义：

> "保留每行 importance 最大的 k 个位置" ⇔ "保留所有 importance ≥ 第 k 大值的位置"

而第 k 大值正好等于 `topk_vals.gather(-1, k_count - 1)` —— 我们已经
有 sorted 后的 topk_vals 了，gather 一次即得。

```python
# 旧：scatter_ 在原坐标上写回 bool 模板
pos = torch.arange(max_keep, device=...)
keep_in_topk = pos < k_count
mask = torch.zeros((B, H, nq, nk), dtype=torch.bool, ...)
mask.scatter_(-1, topk_idx, keep_in_topk)

# 新：elementwise 比较，纯 SIMD，无地址散射
threshold = topk_vals.gather(-1, k_count - 1)   # [B, H, nq, 1]
mask = imp_f >= threshold                        # [B, H, nq, nk]
```

**正确性陷阱（已显式承认）**：
- 当多个位置的 importance 严格相等于 threshold 时，新版会全部保留，
  实际保留块数可能 > `max_keep`。
- 生产里 importance 来自 `softmax(fp32)` 后再 sample/head-mean，
  连续值 ties 概率为 **0**，所以与"严格保留 max_keep 个"在数值上等价。
- 测试里专门构造 uniform tied 用例被改成非均匀输入，避免依赖 tie-break。

**第二轮结果**：scatter_ 从 ms 降到 us，整体降到 ~3 ms。
此时唯一剩下的 ms 级算子就是 `cumsum`（长度 max_keep ≈ 315 仍要 3 ms）。

> 提交：`c8ee4e70 perf(mgm_video): drop scatter_ in ASA block mask via threshold compare`

### 第三步：把 `cumsum` 换成"上三角矩阵 matmul"

**根因复盘**：为什么 cumsum 长度从 451 缩到 315 之后仍然要 3 ms？

NPU 的 `Cumsum` 是**串行前缀扫描算子**：`y[j] = y[j-1] + x[j]`
存在严格数据依赖，不能像 reduce 那样树形并行。即使 max_keep 只有几百，
"行内串行 + 跨行并行度有限 + kernel launch overhead" 也足以让单次
调用维持在毫秒级别。这是硬件结构性短板，**不可能靠该算子自身调优解决**。

**等价改写**：前缀和可以写成矩阵乘。设上三角全 1 矩阵
`M ∈ R^{n×n}`，`M[i, j] = 1 当且仅当 i ≤ j`，则：

```
(x @ M)[j] = Σ_i x[i] · M[i, j] = Σ_{i ≤ j} x[i] = cumsum(x)[j]
```

```python
# 旧：串行扫描
cum = torch.cumsum(topk_vals, dim=-1)

# 新：cube 上的 GEMM
prefix_m = _get_prefix_sum_matrix(max_keep, imp_f.device, imp_f.dtype)
cum = torch.matmul(topk_vals, prefix_m)
```

**为什么 matmul 会快这么多**：
- `[1, 1, 451, 315] @ [315, 315]` ≈ 14M MAC，在 Ascend cube 上是
  ~几十 µs 级别（cube 吞吐与 vector / scalar 相比高出两个数量级）。
- 取代的是无法并行的串行扫描算子，硬件路径完全不同。

**M 的缓存**：M 只取决于 `(n, device, dtype)`，跨 denoise step
完全不变（同一个 shape 的 ASA 计算贯穿全部步），所以做模块级 LRU
缓存即可：

```python
_PREFIX_SUM_MATRIX_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

def _get_prefix_sum_matrix(n, device, dtype):
    key = (device, dtype, n)
    m = _PREFIX_SUM_MATRIX_CACHE.get(key)
    if m is None:
        m = torch.ones((n, n), device=device, dtype=dtype).triu_()
        _PREFIX_SUM_MATRIX_CACHE[key] = m
    return m
```

显存代价：`max_keep × max_keep × 4B`，max_keep=315 时约 **400 KiB**，
对 24/32 GiB NPU 显存可忽略。

**精度**：fp32 GEMM 在 Ascend cube 上以 fp32 累加，n=315 的累加误差
量级约 `n · eps ≈ 3e-5`，远在 fp32 噪声内；下游 `cum < need` 比较的
是 "是否跨过 0.95 × total" 的离散判定，对此量级的误差完全不敏感。

**第三轮结果**：cumsum 3 ms → ~几十 µs，`build_asa_block_mask`
整体进入 **us 级别**。

> 提交：`1f29ec91 perf(mgm_video): replace cumsum with cached upper-tri matmul in ASA mask`

## 5. 优化前后对照

### 最终实现

```python
def build_asa_block_mask(importance, max_retain_ratio, min_retain_ratio, energy_threshold):
    B, H, nq, nk = importance.shape
    imp_f = importance if importance.dtype == torch.float32 else importance.float()

    min_keep = max(1, int(nk * min_retain_ratio))
    max_keep = max(min_keep, int(nk * max_retain_ratio))

    topk_vals, _ = torch.topk(imp_f, max_keep, dim=-1, sorted=True)
    total = imp_f.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    prefix_m = _get_prefix_sum_matrix(max_keep, imp_f.device, imp_f.dtype)
    cum = torch.matmul(topk_vals, prefix_m)   # 等价 cumsum，跑在 cube

    need = energy_threshold * total
    k_count = (cum < need).sum(dim=-1, keepdim=True) + 1
    k_count = k_count.clamp(min=min_keep, max=max_keep)

    threshold = topk_vals.gather(-1, k_count - 1)
    return imp_f >= threshold
```

### 算子映射对照

| 步骤 | 原版（`dfc7620d`） | 终版（`1f29ec91`） | NPU 落地路径 |
|---|---|---|---|
| 选 top-k | `sort` 全 nk | `topk(max_keep)` | vector + cube 辅助 |
| 总能量 | `cum[..., -1:]` 全长 cumsum 切片 | `sum(dim=-1)` 单 reduce | vector reduce |
| 累加和 | `cumsum` 全 nk（串行） | **`@ M`**（上三角 GEMM） | **cube** |
| 找 k | `argmax(over.int()) + any + where` 三连 | `(cum < need).sum() + 1` 单 reduce | vector reduce |
| 回填 mask | `scatter_(-1, indices, keep_sorted)` | `imp_f >= threshold` elementwise | vector SIMD |
| dtype cast | `importance.float()` 无条件 | 仅当非 fp32 时触发 | — |

### 端到端耗时（同一 shape `[1, 1, 451, 451]`，`max_keep ≈ 315`）

| 版本 | 耗时 | 主要瓶颈 |
|---|---|---|
| 原版（`dfc7620d`） | ~8 ms | sort + 全长 cumsum + scatter_ |
| 第一轮（`299c8e82`） | ~3 ms | scatter_（仍在）+ cumsum |
| 第二轮（`c8ee4e70`） | ~3 ms | cumsum（独立瓶颈裸露出来） |
| 终版（`1f29ec91`） | **us 量级** | 主要为 topk + 几个 elementwise |

## 6. 通用经验：NPU 上"PyTorch 一行流"的陷阱

这次优化命中了 NPU 性能调优里的几个普适规律，建议记入个人 checklist：

1. **认 cube，避 vector/scalar 串行算子**：
   - cube（GEMM、conv）单元吞吐远高于 vector / scalar 单元。
   - sort / cumsum / scan / scatter 都是结构性不能上 cube 的算子，
     在 NPU 上即使输入很短也容易达到 ms 量级。
   - 任何能改写成 GEMM 的操作（前缀和、卷积式 reduce、累积乘）都值得改写。

2. **算子长度即代价**：
   - GPU 上"全长 sort + clamp 截断"读起来干净，NPU 上每多 1 倍长度
     就多 1 倍 vector/scalar 时间。下游有 `clamp(max=K)` 时一律先把
     `sort` 改成 `topk(K)`。

3. **Kernel launch overhead 也是 ms 级别**：
   - 单次 launch ~50–100 µs，10+ 个小算子叠起来就是 1–2 ms。
   - `argmax + any + where + cast + add` 这类三连组合优先塌成单 reduce。

4. **`scatter_` 是 NPU 的灾难，能不写就别写**：
   - 散射写入需要逐元素地址解析 + 顺序写，是 NPU 最不友好的访存模式。
   - 大部分场景下都可以反推阈值后用 elementwise 比较替代
     （前提是确认 ties 概率为 0）。

5. **小 shape 的常量缓存收益超出直觉**：
   - 几百 × 几百 的辅助矩阵（mask、index、上三角）只占几百 KB，
     但每步重建会带来不可忽视的开销。模块级缓存是免费午餐。

## 7. 测试与正确性兜底

`tests/mgm_video/test_asa.py` 中保留了 5 个针对 `build_asa_block_mask`
的语义测试，覆盖：

- `dense_probe_keeps_all`：`max_retain=1.0` + 阈值 0.95 时保留全部；
- `energy_threshold_prunes_tail`：能量阈值正确截断尾部；
- `respects_min_retain`：低于 min_keep 时被抬起；
- `respects_max_retain`：超过 max_keep 时被压回；
- `bounds_random`：随机 softmax 输入下行和落入 `[min_keep, max_keep]`。

三轮优化全部 37 个测试通过。其中两个用例（`energy_threshold_prunes_tail`
和 `respects_max_retain`）原本依赖等值输入触发稳定 sort 的 tie-break
行为，被改成显式非均匀输入 —— 这才是符合生产分布的测试方向。

## 8. 后续可能的优化方向

`build_asa_block_mask` 已经降到 us 级别，但整条 ASA 路径上还有
两个潜在热点：

- **`expand_block_to_token_mask` ~40 ms**：当前用 `expand + reshape`
  把 block mask 拍成 token-level dense mask，主要受限于 view/expand
  的物化开销。后续若 SFA 融合算子上线支持 block-level mask 直接喂入
  attention，可以整段省掉。
- **`sample_pool_attn` ~5 ms**：本就是 matmul + softmax 的 cube 任务，
  时间合理；不优先动。

如果未来需要把 `build_asa_block_mask` 进一步压到 100 µs 以内，可以
考虑：
- 把整个函数下沉成单个自定义 NPU kernel（fused topk + 阈值搜索 + 比较），
  消掉 4–5 次 kernel launch overhead；
- 或者在外层把多个 layer 的 importance batch 起来一起处理，摊薄 launch
  成本。

但当前性能下，进一步优化的 ROI 已经很低，建议优先攻 `expand_block_to_token_mask`
那 40 ms 的 SFA 融合。
