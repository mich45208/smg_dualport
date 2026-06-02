use std::sync::{Arc, RwLock};

use tracing::{debug, info};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::{KvEventMonitor, Worker};
use kv_index::compute_request_content_hashes;

/// Configuration for KV cache-centric scheduling policy.
#[derive(Debug, Clone)]
pub struct KvCentricConfig {
    pub kv_bytes_per_token: usize,
    pub block_size: usize,
    pub compute_overhead_ms: f64,
    pub compute_slope_ms: f64,
    pub pd_overhead_ms: f64,
    pub pd_slope_ms_per_mb: f64,
    pub service_time_ms: f64,
    pub balancing_threshold: u32,
}

impl Default for KvCentricConfig {
    fn default() -> Self {
        Self {
            kv_bytes_per_token: 57344,
            block_size: 64,
            compute_overhead_ms: 14.4,
            compute_slope_ms: 0.0098,
            pd_overhead_ms: 2.2,
            pd_slope_ms_per_mb: 0.025,
            service_time_ms: 36.0,
            balancing_threshold: 4,
        }
    }
}

#[derive(Debug)]
struct TtftEstimate {
    ttft_ms: f64,
    queue_ms: f64,
    compute_ms: f64,
    pd_ms: f64,
    cached_blocks: u32,
    new_tokens: usize,
}

#[derive(Debug)]
pub struct KvCentricPolicy {
    config: KvCentricConfig,
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
}

impl KvCentricPolicy {
    pub fn with_config(config: KvCentricConfig) -> Self {
        info!(
            "KvCentricPolicy initialized: kv_bytes_per_token={} block_size={} \
             compute={}ms+{}×tokens pd={}ms+{}×MB queue=depth×{}ms threshold={}",
            config.kv_bytes_per_token,
            config.block_size,
            config.compute_overhead_ms,
            config.compute_slope_ms,
            config.pd_overhead_ms,
            config.pd_slope_ms_per_mb,
            config.service_time_ms,
            config.balancing_threshold,
        );
        Self {
            config,
            kv_monitor: RwLock::new(None),
        }
    }

    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write().unwrap() = monitor;
    }

    fn estimate_ttft(
        &self,
        prompt_tokens: usize,
        local_cached_blocks: u32,
        queue_depth: usize,
    ) -> TtftEstimate {
        let cached_tokens = (local_cached_blocks as usize) * self.config.block_size;
        let new_tokens = prompt_tokens.saturating_sub(cached_tokens);

        let queue_ms = (queue_depth as f64) * self.config.service_time_ms;

        let compute_ms =
            self.config.compute_overhead_ms + self.config.compute_slope_ms * (new_tokens as f64);

        let kv_size_mb =
            (prompt_tokens * self.config.kv_bytes_per_token) as f64 / (1024.0 * 1024.0);
        let pd_ms = self.config.pd_overhead_ms + self.config.pd_slope_ms_per_mb * kv_size_mb;

        let ttft_ms = queue_ms + compute_ms + pd_ms;

        TtftEstimate {
            ttft_ms,
            queue_ms,
            compute_ms,
            pd_ms,
            cached_blocks: local_cached_blocks,
            new_tokens,
        }
    }
}

impl LoadBalancingPolicy for KvCentricPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }
        if healthy_indices.len() == 1 {
            return Some(healthy_indices[0]);
        }

        let tokens = match info.tokens {
            Some(t) if !t.is_empty() => t,
            _ => {
                return healthy_indices
                    .iter()
                    .copied()
                    .min_by_key(|&idx| workers[idx].load());
            }
        };

        let prompt_tokens = tokens.len();

        let monitor_guard = self.kv_monitor.read().unwrap();
        let monitor = match monitor_guard.as_ref() {
            Some(m) => m,
            None => {
                return healthy_indices
                    .iter()
                    .copied()
                    .min_by_key(|&idx| workers[idx].load());
            }
        };

        let model_id = workers[healthy_indices[0]].model_id();
        let indexer = match monitor.get_indexer(model_id) {
            Some(idx) => idx,
            None => {
                return healthy_indices
                    .iter()
                    .copied()
                    .min_by_key(|&idx| workers[idx].load());
            }
        };

        let block_size = monitor
            .block_size(model_id)
            .unwrap_or(self.config.block_size);
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        let overlap = indexer.find_matches(&content_hashes, false);

        let mut best_idx = healthy_indices[0];
        let mut best_ttft = f64::MAX;
        let mut best_estimate: Option<TtftEstimate> = None;

        for &idx in &healthy_indices {
            let worker_url = workers[idx].url();
            let worker_id = indexer.worker_id(worker_url);
            let local_cached = worker_id
                .and_then(|wid| overlap.scores.get(&wid).copied())
                .unwrap_or(0);
            let queue_depth = workers[idx].load();

            let estimate = self.estimate_ttft(prompt_tokens, local_cached, queue_depth);

            if estimate.ttft_ms < best_ttft {
                best_ttft = estimate.ttft_ms;
                best_idx = idx;
                best_estimate = Some(estimate);
            }
        }

        if let Some(est) = &best_estimate {
            debug!(
                "KvCentric: selected worker={} ttft={:.1}ms \
                 (queue={:.1}ms compute={:.1}ms pd={:.1}ms) \
                 cached={} new_tokens={} prompt={}",
                workers[best_idx].url(),
                est.ttft_ms,
                est.queue_ms,
                est.compute_ms,
                est.pd_ms,
                est.cached_blocks,
                est.new_tokens,
                prompt_tokens,
            );
        }

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
