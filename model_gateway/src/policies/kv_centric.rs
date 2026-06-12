use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use kv_index::compute_request_content_hashes;
use serde::Deserialize;
use tracing::debug;
use tracing::info;
use tracing::warn;

use super::LoadBalancingPolicy;
use super::SelectWorkerInfo;
use super::get_healthy_worker_indices;
use crate::worker::KvEventMonitor;
use crate::worker::Worker;
use crate::worker::WorkerType;

/// Env var pointing to a JSON file of coefficient overrides. When set (e.g. to a
/// mounted k8s ConfigMap path), the values in that file override the compiled /
/// CLI defaults at startup — letting the oracle be retuned WITHOUT recompiling.
pub const OVERRIDE_ENV: &str = "SMG_KV_CENTRIC_CONFIG";

/// Default calibrated coefficients (Qwen2.5-7B, GB300) — SINGLE SOURCE OF TRUTH.
///
/// `config/types.rs` (serde defaults), `main.rs` (CLI defaults), and the Python
/// bindings (`bindings/python/src/lib.rs`) all reference these constants so the
/// defaults cannot silently drift across config entry-paths. The Python dataclass
/// in `router_args.py` keeps a mirror (it cannot import Rust consts) and is marked
/// to be kept in sync with this module.
pub mod defaults {
    pub const KV_BYTES_PER_TOKEN: usize = 57344;
    pub const COMPUTE_OVERHEAD_MS: f64 = 14.8;
    pub const COMPUTE_SLOPE_MS: f64 = 0.00464;
    pub const COMPUTE_QUAD_MS: f64 = 4.75e-7;
    pub const LOAD_OVERHEAD_MS: f64 = 14.9;
    pub const L3_READ_OVERHEAD_MS: f64 = 0.215;
    pub const L3_READ_PER_TOKEN_MS: f64 = 0.000937;
    pub const BALANCING_THRESHOLD: u32 = 4;
}

/// Configuration for the KV cache-centric scheduling policy.
///
/// Picks the prefill worker that minimizes estimated time-to-first-token:
///     TTFT = T_compute(c, n_uncached) + T_l3_read(n_pulled)
///
/// where `c` is the worker's concurrency (in-flight requests + this one) and
/// `n_uncached` is the prompt length minus the prefix already cached. The
/// queue wait is folded into `T_compute` (see below) rather than modeled
/// separately, because under continuous batching the contention cost shows up
/// in the forward pass, not in a separate queue.
///
/// All coefficients are empirically calibrated (Qwen2.5-7B, GB300). Defaults
/// are documented per field; override per-deployment via CLI / PolicyConfig.
#[derive(Debug, Clone)]
pub struct KvCentricConfig {
    /// KV cache bytes per token (Qwen2.5-7B = 57344). Used for L3 size logging.
    pub kv_bytes_per_token: usize,
    /// Fallback block size if the KV-event monitor hasn't learned one yet.
    pub block_size: usize,

    // --- Compute model (queue + prefill, load-aware) ---
    // T_compute(c, n) = max( isolated, serialized )
    //   isolated   = compute_overhead_ms + compute_slope_ms*n + compute_quad_ms*n^2
    //   serialized = load_overhead_ms    + compute_slope_ms*c*n
    // Low load -> isolated (attention-bound quadratic); under load -> serialized
    // (per-token cost scales ~linearly with concurrency). R^2=0.99, mean err 11%.
    /// Isolated single-request overhead (ms). Default 14.8.
    pub compute_overhead_ms: f64,
    /// Per-token compute slope (ms/token), also the per-concurrency scaling. Default 0.00464.
    pub compute_slope_ms: f64,
    /// Attention quadratic term (ms/token^2), single-request only. Default 4.75e-7.
    pub compute_quad_ms: f64,
    /// Serialized-batch overhead (ms) under load. Default 14.9.
    pub load_overhead_ms: f64,

    // --- L3 read model (Mooncake store -> prefill GPU, remote RDMA) ---
    // T_l3_read(n) = l3_read_overhead_ms + l3_read_per_token_ms * n
    // ~57 GiB/s, R^2=0.983, flat to 16 concurrent pulls. PD write is async (excluded).
    /// Per-pull fixed overhead (ms, master RPC). Default 0.215.
    pub l3_read_overhead_ms: f64,
    /// L3 read cost per cached token pulled (ms/token). Default 0.000937 (~0.94us/tok).
    pub l3_read_per_token_ms: f64,

    /// Only consider pulling a remote prefix from L3 when it exceeds the local
    /// match by more than this many blocks (avoids churn on tiny diffs). Default 4.
    pub balancing_threshold: u32,
}

impl Default for KvCentricConfig {
    fn default() -> Self {
        Self {
            kv_bytes_per_token: defaults::KV_BYTES_PER_TOKEN,
            block_size: 64,
            compute_overhead_ms: defaults::COMPUTE_OVERHEAD_MS,
            compute_slope_ms: defaults::COMPUTE_SLOPE_MS,
            compute_quad_ms: defaults::COMPUTE_QUAD_MS,
            load_overhead_ms: defaults::LOAD_OVERHEAD_MS,
            l3_read_overhead_ms: defaults::L3_READ_OVERHEAD_MS,
            l3_read_per_token_ms: defaults::L3_READ_PER_TOKEN_MS,
            balancing_threshold: defaults::BALANCING_THRESHOLD,
        }
    }
}

/// Partial coefficient overrides loadable from a JSON file at runtime. Every field
/// is optional: only the keys present in the JSON override the existing config; the
/// rest are left untouched. This lets the oracle be retuned WITHOUT recompiling —
/// point `SMG_KV_CENTRIC_CONFIG` at a JSON file (e.g. a mounted k8s ConfigMap).
///
/// Example JSON (any subset):
/// ```json
/// { "compute_slope_ms": 0.005, "l3_read_per_token_ms": 0.0011, "balancing_threshold": 8 }
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvCentricOverrides {
    pub kv_bytes_per_token: Option<usize>,
    pub block_size: Option<usize>,
    pub compute_overhead_ms: Option<f64>,
    pub compute_slope_ms: Option<f64>,
    pub compute_quad_ms: Option<f64>,
    pub load_overhead_ms: Option<f64>,
    pub l3_read_overhead_ms: Option<f64>,
    pub l3_read_per_token_ms: Option<f64>,
    pub balancing_threshold: Option<u32>,
}

impl KvCentricConfig {
    /// Apply partial overrides; only `Some(_)` fields take effect.
    pub fn apply_overrides(&mut self, o: &KvCentricOverrides) {
        if let Some(v) = o.kv_bytes_per_token {
            self.kv_bytes_per_token = v;
        }
        if let Some(v) = o.block_size {
            self.block_size = v;
        }
        if let Some(v) = o.compute_overhead_ms {
            self.compute_overhead_ms = v;
        }
        if let Some(v) = o.compute_slope_ms {
            self.compute_slope_ms = v;
        }
        if let Some(v) = o.compute_quad_ms {
            self.compute_quad_ms = v;
        }
        if let Some(v) = o.load_overhead_ms {
            self.load_overhead_ms = v;
        }
        if let Some(v) = o.l3_read_overhead_ms {
            self.l3_read_overhead_ms = v;
        }
        if let Some(v) = o.l3_read_per_token_ms {
            self.l3_read_per_token_ms = v;
        }
        if let Some(v) = o.balancing_threshold {
            self.balancing_threshold = v;
        }
    }

    /// Load JSON overrides from `path` and apply them (fail-soft: read/parse errors
    /// are logged and the config is left unchanged).
    pub fn apply_overrides_from_file(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<KvCentricOverrides>(&content) {
                Ok(overrides) => {
                    self.apply_overrides(&overrides);
                    info!("KvCentric: applied coefficient overrides from {path}");
                }
                Err(e) => {
                    warn!("KvCentric: failed to parse override file {path}: {e} (keeping defaults)")
                }
            },
            Err(e) => {
                warn!("KvCentric: failed to read override file {path}: {e} (keeping defaults)")
            }
        }
    }

    /// If `SMG_KV_CENTRIC_CONFIG` is set to a non-empty path, load & apply overrides.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(path) = std::env::var(OVERRIDE_ENV) {
            if !path.trim().is_empty() {
                self.apply_overrides_from_file(path.trim());
            }
        }
    }
}

#[derive(Debug)]
struct TtftEstimate {
    ttft_ms: f64,
    compute_ms: f64,
    transfer_ms: f64,
    concurrency: usize,
    local_cached_blocks: u32,
    pulled_tokens: usize,
    new_tokens: usize,
}

#[derive(Debug)]
pub struct KvCentricPolicy {
    config: KvCentricConfig,
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
    /// Serializes the decision-and-reserve critical section in `select_worker` so
    /// that concurrent selections observe each other's prefill in-flight reservation.
    /// Without this, a simultaneous burst all reads `prefill_inflight()==0` before any
    /// reservation lands and herds onto the cache owner (cold-start concentration).
    /// The expensive cache-overlap hashing is done OUTSIDE this lock; the locked
    /// section is only N atomic reads + argmin + one increment (sub-microsecond).
    select_lock: Mutex<()>,
}

impl KvCentricPolicy {
    pub fn with_config(config: KvCentricConfig) -> Self {
        info!(
            "KvCentricPolicy initialized: kv_bytes_per_token={} block_size={} \
             compute=max({}+{}*n+{:e}*n^2, {}+{}*c*n)ms l3_read={}+{}*n_cached ms threshold={}",
            config.kv_bytes_per_token,
            config.block_size,
            config.compute_overhead_ms,
            config.compute_slope_ms,
            config.compute_quad_ms,
            config.load_overhead_ms,
            config.compute_slope_ms,
            config.l3_read_overhead_ms,
            config.l3_read_per_token_ms,
            config.balancing_threshold,
        );
        Self {
            config,
            kv_monitor: RwLock::new(None),
            select_lock: Mutex::new(()),
        }
    }

    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write().unwrap() = monitor;
    }

    /// Reserve a prefill in-flight slot on the chosen worker, IF it performs prefill
    /// compute — i.e. a Prefill worker in PD mode, or a Regular worker in a
    /// non-disaggregated deployment. Decode-pool selections in PD mode must NOT reserve
    /// a prefill slot, so Decode workers are skipped. The matching RELEASE is the
    /// `PrefillReservationGuard` the grpc worker-selection stage creates for the chosen
    /// prefill worker. Callers reserving from a multi-candidate decision hold
    /// `select_lock` so the read→reserve is atomic across concurrent selections.
    fn reserve_compute_slot(&self, workers: &[Arc<dyn Worker>], idx: usize) {
        if !matches!(workers[idx].worker_type(), WorkerType::Decode) {
            workers[idx].increment_prefill_inflight();
        }
    }

    /// Load-aware compute time: max of the isolated (attention-bound) cost and
    /// the serialized-batch cost (per-token scaled by concurrency). Queue wait
    /// is absorbed here.
    fn t_compute(&self, concurrency: usize, new_tokens: usize) -> f64 {
        let n = new_tokens as f64;
        let c = concurrency.max(1) as f64;
        let isolated = self.config.compute_overhead_ms
            + self.config.compute_slope_ms * n
            + self.config.compute_quad_ms * n * n;
        let serialized = self.config.load_overhead_ms + self.config.compute_slope_ms * c * n;
        isolated.max(serialized)
    }

    /// L3 read (remote RDMA pull) time for `cached_tokens` pulled from the store.
    fn t_l3_read(&self, cached_tokens: usize) -> f64 {
        if cached_tokens == 0 {
            return 0.0;
        }
        self.config.l3_read_overhead_ms + self.config.l3_read_per_token_ms * (cached_tokens as f64)
    }

    /// Estimate TTFT for routing `prompt_tokens` to a worker that has
    /// `local_cached_blocks` of the prefix locally, given the cluster-best match
    /// is `best_cached_blocks` (available via L3 write-through) and the worker
    /// currently has `queue_depth` in-flight requests.
    fn estimate_ttft(
        &self,
        prompt_tokens: usize,
        local_cached_blocks: u32,
        best_cached_blocks: u32,
        queue_depth: usize,
        block_size: usize,
    ) -> TtftEstimate {
        // This request joins the running batch, so effective concurrency = depth + 1.
        let concurrency = queue_depth + 1;
        let local_cached_tokens = (local_cached_blocks as usize) * block_size;

        // Option A: use only the local prefix, recompute the rest (no L3 pull).
        let new_a = prompt_tokens.saturating_sub(local_cached_tokens);
        let compute_a = self.t_compute(concurrency, new_a);
        let mut best = TtftEstimate {
            ttft_ms: compute_a,
            compute_ms: compute_a,
            transfer_ms: 0.0,
            concurrency,
            local_cached_blocks,
            pulled_tokens: 0,
            new_tokens: new_a,
        };

        // Option B: pull the extra cached prefix (best - local) from L3, recompute
        // only the remainder. Worth it when the transfer is cheaper than recomputing
        // those tokens. Gated by balancing_threshold to avoid churn on tiny diffs.
        if best_cached_blocks > local_cached_blocks + self.config.balancing_threshold {
            let best_cached_tokens = (best_cached_blocks as usize) * block_size;
            let pull_tokens = best_cached_tokens.saturating_sub(local_cached_tokens);
            let new_b = prompt_tokens.saturating_sub(best_cached_tokens);
            let compute_b = self.t_compute(concurrency, new_b);
            let transfer_b = self.t_l3_read(pull_tokens);
            let ttft_b = compute_b + transfer_b;
            if ttft_b < best.ttft_ms {
                best = TtftEstimate {
                    ttft_ms: ttft_b,
                    compute_ms: compute_b,
                    transfer_ms: transfer_b,
                    concurrency,
                    local_cached_blocks,
                    pulled_tokens: pull_tokens,
                    new_tokens: new_b,
                };
            }
        }

        best
    }
}

/// Per-worker cache overlap snapshot, computed (lock-free) before the reservation
/// critical section. `local_cached[i]` aligns with the i-th healthy worker index.
struct CacheModel {
    prompt_tokens: usize,
    block_size: usize,
    local_cached: Vec<u32>,
    best_cached_blocks: u32,
}

impl LoadBalancingPolicy for KvCentricPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        // Single candidate: no decision, no race — just reserve and return.
        if healthy_indices.len() == 1 {
            let idx = healthy_indices[0];
            self.reserve_compute_slot(workers, idx);
            return Some(idx);
        }

        // ---- Cache overlap (read-only w.r.t. reservation state) — computed WITHOUT the
        // selection lock so the expensive hashing/lookup never serializes. Yields None
        // when the cost model can't apply (no tokens / monitor / indexer); we then fall
        // back to least-loaded. The kv_monitor read lock is released at the end of this
        // block, before select_lock is acquired (no nested locking).
        let cache: Option<CacheModel> = (|| {
            let tokens = match info.tokens {
                Some(t) if !t.is_empty() => t,
                _ => return None,
            };
            let monitor_guard = self.kv_monitor.read().unwrap();
            let monitor = monitor_guard.as_ref()?;
            let model_id = workers[healthy_indices[0]].model_id();
            let indexer = monitor.get_indexer(model_id)?;
            let block_size = monitor
                .block_size(model_id)
                .unwrap_or(self.config.block_size);
            let content_hashes = compute_request_content_hashes(tokens, block_size);
            let overlap = indexer.find_matches(&content_hashes, false);

            // Per-worker local match + cluster-best match (the best is reachable via L3
            // write-through, so any worker can in principle pull up to best_cached).
            let mut local_cached: Vec<u32> = Vec::with_capacity(healthy_indices.len());
            let mut best_cached_blocks = 0u32;
            for &idx in &healthy_indices {
                let cached = indexer
                    .worker_id(workers[idx].url())
                    .and_then(|wid| overlap.scores.get(&wid).copied())
                    .unwrap_or(0);
                best_cached_blocks = best_cached_blocks.max(cached);
                local_cached.push(cached);
            }
            Some(CacheModel {
                prompt_tokens: tokens.len(),
                block_size,
                local_cached,
                best_cached_blocks,
            })
        })();

        // ---- Decision + reservation critical section. Serialized so a concurrent burst
        // observes each prior reservation, instead of every selection reading
        // prefill_inflight()==0 and herding onto the cache owner. The body is only N
        // atomic reads + argmin + one increment (sub-microsecond); the hashing above ran
        // outside this lock.
        let _sel = self.select_lock.lock().unwrap();

        let best_pos = match &cache {
            Some(cm) => {
                let mut best_pos = 0usize;
                let mut best_ttft = f64::MAX;
                let mut best_estimate: Option<TtftEstimate> = None;
                for (pos, &idx) in healthy_indices.iter().enumerate() {
                    // queue depth = requests reserved on this prefill that haven't left
                    // prefill yet (queue + executing). Reserved here, released at drop.
                    let queue_depth = workers[idx].prefill_inflight();
                    let estimate = self.estimate_ttft(
                        cm.prompt_tokens,
                        cm.local_cached[pos],
                        cm.best_cached_blocks,
                        queue_depth,
                        cm.block_size,
                    );
                    if estimate.ttft_ms < best_ttft {
                        best_ttft = estimate.ttft_ms;
                        best_pos = pos;
                        best_estimate = Some(estimate);
                    }
                }
                if let Some(est) = &best_estimate {
                    debug!(
                        "KvCentric: selected worker={} ttft={:.1}ms \
                         (compute={:.1}ms transfer={:.1}ms c={}) \
                         local_cached={}blk pulled={}tok new={}tok prompt={}tok",
                        workers[healthy_indices[best_pos]].url(),
                        est.ttft_ms,
                        est.compute_ms,
                        est.transfer_ms,
                        est.concurrency,
                        est.local_cached_blocks,
                        est.pulled_tokens,
                        est.new_tokens,
                        cm.prompt_tokens,
                    );
                }
                best_pos
            }
            None => {
                // Fallback: least prefill-loaded worker (still under the lock so the
                // read→reserve stays atomic).
                let mut best_pos = 0usize;
                let mut best_load = usize::MAX;
                for (pos, &idx) in healthy_indices.iter().enumerate() {
                    let load = workers[idx].prefill_inflight();
                    if load < best_load {
                        best_load = load;
                        best_pos = pos;
                    }
                }
                best_pos
            }
        };

        let best_idx = healthy_indices[best_pos];
        self.reserve_compute_slot(workers, best_idx);
        Some(best_idx)
    }

    fn name(&self) -> &'static str {
        "kv_centric"
    }

    fn needs_request_text(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
