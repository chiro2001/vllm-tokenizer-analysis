// SPDX-License-Identifier: Apache-2.0
//! 计时：warmup + 多轮采样 + 中位数。
//!
//! 为什么自建而不用 criterion 跑全部维度：criterion 的自适应采样在
//! 「6 后端 × 4 语料 × 3 长度 × 4 操作 × 2 线程」的笛卡尔积上会跑几小时，
//! 而本项目的时间盒不允许。criterion 仍保留在 `benches/backends.rs`，
//! 用于**少量关键组合**的交叉验证（两种装置的数应当落在彼此 10% 以内）。
//!
//! 统计口径（写进文档、不要只给数字）：
//! - 每次 `op` 调用一次 [`Instant::now()`] 前后差；
//! - `iters` 次取中位数（不是平均），抗调度抖动；
//! - 全部迭代在同一个线程上跑（`taskset` 已在 `scripts/limit.sh` 里绑核）；
//! - 报告单位：µs/op（encode/decode 单次），以及派生的 MB/s、tokens/s。

use std::time::{Duration, Instant};

use serde::Serialize;

/// 与 `anyhow::Result` 兼容的错误类型别名（harness 内部统一用它）。
pub type MeasureResult<T> = Result<T, String>;

/// 在**隔离的 panic 边界**里跑一段代码。
///
/// 为什么需要：被测的后端是第三方 crate，某个输入可能让它 panic
/// （实测 `fastokens 0.2.1` 的长中文输入就会，见 `docs/03-backend-matrix.md`）。
/// 整轮矩阵不能因为一个输入就死掉，所以每个测量点都套一层 `catch_unwind`：
/// panic 被记成该点的失败行，其余点照跑。
///
/// 前提：本 harness 的 release profile **保留 `panic = unwind`**
/// （上游 vLLM 是 `panic = "abort"`，这里有意偏离，理由见 Cargo.toml 注释），
/// 否则 `catch_unwind` 不生效。
pub fn guard_panic<T>(what: &str, f: impl FnOnce() -> T + std::panic::UnwindSafe) -> MeasureResult<T> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // 静音：错误信息由我们带回结果
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    result.map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(非字符串 panic payload)".to_string());
        format!("panic in {what}: {msg}")
    })
}

/// 一次测量的结果。
#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    /// 中位数耗时，µs。
    pub median_us: f64,
    /// 最小耗时，µs（最接近"无干扰"的一轮）。
    pub min_us: f64,
    /// 最大耗时，µs。
    pub max_us: f64,
    /// p90 耗时，µs。
    pub p90_us: f64,
    pub iters: usize,
    /// 每次迭代覆盖的输入字节数（用于 MB/s）。
    pub input_bytes: usize,
    /// 每次迭代覆盖的 token 数（用于 tok/s）。
    pub input_tokens: usize,
}

impl Measurement {
    pub fn mb_per_s(&self) -> f64 {
        if self.median_us <= 0.0 {
            return f64::NAN;
        }
        (self.input_bytes as f64 / (self.median_us / 1e6)) / 1e6
    }

    pub fn tokens_per_s(&self) -> f64 {
        if self.median_us <= 0.0 {
            return f64::NAN;
        }
        self.input_tokens as f64 / (self.median_us / 1e6)
    }

    pub fn us_per_token(&self) -> f64 {
        if self.input_tokens == 0 {
            return f64::NAN;
        }
        self.median_us / self.input_tokens as f64
    }
}

/// 计时器。
pub struct Timer {
    warmup: usize,
    iters: usize,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new(3, 20)
    }
}

impl Timer {
    /// `warmup` 轮不计时（把缓存、分支预测、分配器预热好），
    /// `iters` 轮计时。
    pub fn new(warmup: usize, iters: usize) -> Self {
        Self { warmup: warmup.max(1), iters: iters.max(1) }
    }

    pub fn iters(&self) -> usize {
        self.iters
    }

    /// 跑一次测量。`f` 每轮调用一次；返回值被 `black_box`（防止被优化掉）。
    pub fn measure<R>(
        &self,
        input_bytes: usize,
        input_tokens: usize,
        mut f: impl FnMut() -> R,
    ) -> Measurement {
        for _ in 0..self.warmup {
            std::hint::black_box(f());
        }
        let mut samples: Vec<f64> = Vec::with_capacity(self.iters);
        for _ in 0..self.iters {
            let t0 = Instant::now();
            std::hint::black_box(f());
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Measurement::from_samples(samples, input_bytes, input_tokens)
    }

    /// 自适应版：先跑 `probe` 轮估计单轮耗时，把总测量时间压到
    /// `budget_ms` 以内（最少 3 轮，最多 `max_iters` 轮）。
    ///
    /// 8k 输入的单次 encode 在慢后端上可能到毫秒级，固定 20 轮会让
    /// 整个矩阵跑不完；在线路径关心的是**单次延迟**，轮数少一点没关系，
    /// 我们的判据是中位数的稳定性（见文档的"噪声地板"一节）。
    pub fn measure_with_budget<R>(
        &self,
        input_bytes: usize,
        input_tokens: usize,
        budget_ms: f64,
        max_iters: usize,
        mut f: impl FnMut() -> R,
    ) -> Measurement {
        for _ in 0..self.warmup {
            std::hint::black_box(f());
        }
        let t_probe = Instant::now();
        std::hint::black_box(f());
        let probe = t_probe.elapsed().as_secs_f64() * 1e3; // ms
        let iters = if probe <= 0.0 {
            max_iters
        } else {
            ((budget_ms / probe).floor() as usize).clamp(3, max_iters.max(3))
        };
        let mut samples: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            std::hint::black_box(f());
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Measurement::from_samples(samples, input_bytes, input_tokens)
    }

    /// 同 [`Self::measure_with_budget`]，但被测闭包可以失败。
    ///
    /// 失败**不 panic**：某个后端在某个输入上出错时（例如 tekken 对不完整
    /// UTF-8 直接报错），harness 要继续测其它后端，并把错误原样记进产物。
    pub fn measure_try_with_budget<T>(
        &self,
        input_bytes: usize,
        input_tokens: usize,
        budget_ms: f64,
        max_iters: usize,
        mut f: impl FnMut() -> MeasureResult<T>,
    ) -> MeasureResult<Measurement> {
        for _ in 0..self.warmup {
            let _ = f()?;
        }
        let t_probe = Instant::now();
        f()?;
        let probe = t_probe.elapsed().as_secs_f64() * 1e3;
        let iters = if probe <= 0.0 {
            max_iters
        } else {
            ((budget_ms / probe).floor() as usize).clamp(3, max_iters.max(3))
        };
        let mut samples: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f()?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(Measurement::from_samples(samples, input_bytes, input_tokens))
    }
}

impl Measurement {
    fn from_samples(mut samples: Vec<f64>, input_bytes: usize, input_tokens: usize) -> Self {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let median = if n % 2 == 1 {
            samples[n / 2]
        } else {
            (samples[n / 2 - 1] + samples[n / 2]) / 2.0
        };
        let p90_idx = ((n as f64 * 0.9).ceil() as usize).saturating_sub(1).min(n - 1);
        Self {
            median_us: median,
            min_us: samples[0],
            max_us: samples[n - 1],
            p90_us: samples[p90_idx],
            iters: n,
            input_bytes,
            input_tokens,
        }
    }
}

/// 忙等一小会儿，让上一轮的热量散掉（跨后端切换时用）。
pub fn settle() {
    std::thread::sleep(Duration::from_millis(50));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_samples() {
        let m = Measurement::from_samples(vec![1.0, 3.0, 2.0], 100, 10);
        assert_eq!(m.median_us, 2.0);
        assert_eq!(m.min_us, 1.0);
        assert_eq!(m.p90_us, 3.0);
    }

    #[test]
    fn throughput_units() {
        let m = Measurement::from_samples(vec![1000.0], 1_000_000, 1000);
        assert!((m.mb_per_s() - 1000.0).abs() < 1e-6);
        assert!((m.tokens_per_s() - 1_000_000.0).abs() < 1e-6);
    }
}
