# 推理速度与频率换算

具身系统需要区分模型调用耗时、batch throughput、低层控制频率和策略重规划频率。本方法提供通用 callable benchmark 与 latency↔Hz 接口，不假设具体策略模型、观测结构或推理框架。

实现位于 `Embodied/inference_timing/`。

## 核心定义

```text
inference_frequency_hz = 1 / mean_inference_latency_seconds
samples_per_second     = batch_size * inference_frequency_hz
replanning_hz          = control_frequency_hz / executed_actions_per_inference
synchronous_replanning_hz = 1 / (inference_seconds + executed_actions / control_hz)
overlapped_capacity_hz     = min(inference_frequency_hz, replanning_hz)
```

- `inference_frequency_hz` 是单线程顺序模型调用次数/秒；
- `samples_per_second` 是 batch 样本吞吐，不能写成策略控制 Hz；
- `control_frequency_hz` 是底层动作执行频率；
- `replanning_hz` 还取决于每次推理后实际执行多少个 action；
- `synchronous_replanning_hz` 对应先推理、再执行整段 action 的串行上限；
- `overlapped_capacity_hz` 仅对应推理与执行完全重叠的容量上界，不是实测闭环频率。

## 直接换算

```python
from Embodied.inference_timing import convert_hz_to_latency, convert_latency_to_hz

frequency_hz = convert_latency_to_hz(40.0, unit="milliseconds")  # 25 Hz
latency_ms = convert_hz_to_latency(25.0, unit="milliseconds")    # 40 ms
```

`unit` 支持 `"seconds"`、`"milliseconds"`、`"microseconds"`，输入必须为正 finite 标量。

## CPU/同步模型迁移

```python
from Embodied.inference_timing import benchmark_inference

# 只测 model forward：预处理提前完成。
model.eval()

model_inputs = prepare_inputs(observation)

result = benchmark_inference(
    model,
    args=(model_inputs,),
    warmup_runs=10,
    timed_runs=100,
    batch_size=model_inputs.shape[0],
    timing_scope="model_forward",
)

print(result.mean_latency_milliseconds)
print(result.p95_latency_seconds * 1000)
print(result.inference_frequency_hz)
print(result.samples_per_second)
```

如果目标是端到端 latency，把预处理、模型和后处理全部放进被计时 callable：

```python
def end_to_end_policy_call(raw_observation):
    inputs = preprocess(raw_observation)
    outputs = model(inputs)
    return decode_actions(outputs)

result = benchmark_inference(
    end_to_end_policy_call,
    args=(observation,),
    warmup_runs=10,
    timed_runs=100,
    timing_scope="observation_to_action",
)
```

模型-only 与 end-to-end 是两种不同指标，必须分别命名和报告，不能在 baseline/ablation 之间切换边界。

## CUDA/异步设备迁移

CUDA kernel 默认异步执行。如果只在 Python 调用前后读时钟而不 synchronize，得到的主要是 launch 时间。

```python
import torch

from Embodied.inference_timing import (
    benchmark_inference,
    create_torch_device_synchronizer,
)

device = torch.device("cuda:0")
synchronize = create_torch_device_synchronizer(device)

result = benchmark_inference(
    model,
    args=(model_inputs,),
    warmup_runs=20,
    timed_runs=200,
    batch_size=model_inputs.shape[0],
    synchronize=synchronize,
)
```

其他异步 runtime 可以传入自己的无参数 `synchronize()` callback。同步发生在 warmup 后，以及每个 timed call 的开始前和结束后。

## Action chunk 与重规划频率

```python
from Embodied.inference_timing import (
    compute_overlapped_replanning_capacity_hz,
    compute_synchronous_replanning_hz,
    convert_control_frequency_to_replanning_hz,
)

action_limited_hz = convert_control_frequency_to_replanning_hz(
    control_frequency_hz=20.0,
    executed_actions_per_inference=4,
)  # 5 Hz

synchronous_hz = compute_synchronous_replanning_hz(
    inference_latency=result.mean_latency_seconds,
    control_frequency_hz=20.0,
    executed_actions_per_inference=4,
)

overlapped_capacity_hz = compute_overlapped_replanning_capacity_hz(
    inference_frequency_hz=result.inference_frequency_hz,
    control_frequency_hz=20.0,
    executed_actions_per_inference=4,
)
```

如果策略预测 16 步但每次只执行 4 步，应传 4 而不是 16。真实系统还受观测采集、通信、排队、后处理和 actuator delay 限制；最终闭环频率应从系统时间戳另外实测。

## 统计结果

`InferenceLatencyResult` 保留所有 `durations_seconds`，并提供：

- mean、median/P50、minimum、maximum、population standard deviation；
- P90、P95、P99 线性插值延迟；
- mean latency 推导的 `inference_frequency_hz`；
- `batch_size * inference_frequency_hz` 得到的 `samples_per_second`。

不要从 mean of per-run Hz 计算最终频率；应先求 mean latency，再取倒数。尾延迟对闭环控制很重要，至少同时报告 median/P95/P99。

## 公平比较与报告

- 固定硬件、device、dtype、batch size、线程数、功耗模式和软件版本。
- 固定输入 shape、token/image 数、action horizon、cache 状态和编译/量化设置。
- 明确是否包含数据搬运、预处理、采样循环、解码、后处理与同步。
- 所有实验组使用相同 warmup/timed run 数与输入序列；不要只给新方法额外 warmup。
- 首次编译、CUDA graph capture、kernel autotune 和 cache 建立通常属于 warmup，但若部署每次都发生则必须计入端到端指标。
- 生成/扩散策略应固定迭代步数；early exit 或可变输出长度需按实际长度分层报告。
- 记录原始 durations，而不只保存一个 Hz，方便统一重算统计量。

## 已知限制

- `time.perf_counter_ns` 测量 wall-clock callable latency，不替代 profiler 或设备 event 的 kernel 级分析。
- Python 循环开销会进入极短 callable 的结果；此时应增加单次调用工作量或使用设备 event 交叉验证。
- benchmark 丢弃返回值且不测结果消费成本；异步执行必须提供正确 synchronization。
- 两种 replanning 接口都是理想上限，不模拟多线程队列、deadline miss、observation staleness 或 jitter；真实 achieved Hz 必须从系统时间戳计算。
