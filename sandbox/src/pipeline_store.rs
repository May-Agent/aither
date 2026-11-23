//! Per-stage buffers, disk spill, active-duration retention, and expiry for
//! tapped shell pipelines.

use std::collections::{HashMap, HashSet};
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_lite::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::LaunchId;
use crate::job_registry::{JobDelta, JobRead, JobStatus, TerminalStream};
use crate::output_compress::render_terminal_bytes;
use crate::pipeline::{
    PipelineReadError, PipelineSnapshot, PipelineStageInfo, StageStatus, StageStreamSelector,
    StageUiDelta, strip_sgr_sequences,
};

/// Bytes buffered per stage stream in memory before spilling to disk.
const STAGE_MEMORY_CAP: usize = 256 * 1024;

/// Spillable byte stream stored in memory up to [`STAGE_MEMORY_CAP`] and on
/// disk thereafter.
#[derive(Debug, Default)]
pub struct SpillStream {
    mem: Vec<u8>,
    file_path: Option<PathBuf>,
    file: Option<async_fs::File>,
    total_bytes: usize,
}

impl SpillStream {
    const fn len(&self) -> usize {
        self.total_bytes
    }

    async fn append(&mut self, spill_dir: Option<&Path>, name: &str, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if self.file.is_none()
            && self.mem.len().saturating_add(bytes.len()) > STAGE_MEMORY_CAP
            && let Some(dir) = spill_dir
            && async_fs::create_dir_all(dir).await.is_ok()
        {
            let path = dir.join(name);
            if let Ok(mut file) = async_fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .await
                && file.write_all(&self.mem).await.is_ok()
            {
                self.mem.clear();
                self.mem.shrink_to_fit();
                self.file_path = Some(path);
                self.file = Some(file);
            }
        }
        if let Some(file) = self.file.as_mut() {
            if file.write_all(bytes).await.is_ok() {
                self.total_bytes += bytes.len();
                return;
            }
            self.file = None;
        }
        self.mem.extend_from_slice(bytes);
        self.total_bytes += bytes.len();
    }

    async fn close_writer(&mut self) {
        if let Some(mut file) = self.file.take() {
            let _ = file.flush().await;
        }
    }

    async fn clear(&mut self) {
        self.close_writer().await;
        self.mem.clear();
        self.mem.shrink_to_fit();
        self.total_bytes = 0;
        if let Some(path) = self.file_path.take() {
            let _ = async_fs::remove_file(path).await;
        }
    }

    async fn read_slice(&self, cursor: usize, max_bytes: usize) -> Result<Vec<u8>, String> {
        if cursor >= self.total_bytes || max_bytes == 0 {
            return Ok(Vec::new());
        }
        let to_read = (self.total_bytes - cursor).min(max_bytes);
        if let Some(path) = &self.file_path {
            let mut file = async_fs::File::open(path)
                .await
                .map_err(|e| format!("open {}: {e}", path.display()))?;
            file.seek(SeekFrom::Start(cursor as u64))
                .await
                .map_err(|e| format!("seek {}: {e}", path.display()))?;
            let mut buf = vec![0_u8; to_read];
            let mut read_total = 0;
            while read_total < to_read {
                let n = file
                    .read(&mut buf[read_total..])
                    .await
                    .map_err(|e| format!("read {}: {e}", path.display()))?;
                if n == 0 {
                    break;
                }
                read_total += n;
            }
            buf.truncate(read_total);
            return Ok(buf);
        }
        Ok(self.mem[cursor..cursor + to_read].to_vec())
    }

    async fn read_all(&self) -> Result<Vec<u8>, String> {
        self.read_slice(0, self.total_bytes).await
    }
}

/// Mutable state for one `0`-indexed stage of a pipeline.
#[derive(Debug)]
pub struct StageRecord {
    pub step: usize,
    pub command: String,
    pub status: StageStatus,
    pub own_duration: Option<Duration>,
    pub retained: bool,
    pub aliased_final: bool,
    pub ui: SpillStream,
    pub native_combined: SpillStream,
    pub native_stdout: SpillStream,
    pub native_stderr: SpillStream,
    pub total_lines: usize,
}

impl StageRecord {
    fn new(step: usize, command: String) -> Self {
        Self {
            step,
            command,
            status: StageStatus::Pending,
            own_duration: None,
            retained: true,
            aliased_final: false,
            ui: SpillStream::default(),
            native_combined: SpillStream::default(),
            native_stdout: SpillStream::default(),
            native_stderr: SpillStream::default(),
            total_lines: 0,
        }
    }

    fn info(&self) -> PipelineStageInfo {
        PipelineStageInfo {
            step: self.step,
            command: self.command.clone(),
            status: self.status,
            own_duration: self.own_duration,
            retained: self.retained,
            stdout_bytes: self.native_stdout.len(),
            stderr_bytes: self.native_stderr.len(),
            ui_bytes: self.ui.len(),
            total_lines: self.total_lines,
        }
    }

    async fn flush_writers(&mut self) {
        self.ui.close_writer().await;
        self.native_combined.close_writer().await;
        self.native_stdout.close_writer().await;
        self.native_stderr.close_writer().await;
    }

    async fn drop_buffers(&mut self) {
        self.ui.clear().await;
        self.native_combined.clear().await;
        self.native_stdout.clear().await;
        self.native_stderr.clear().await;
    }
}

/// Active or completed pipeline attached to a job.
#[derive(Debug)]
pub struct LivePipeline {
    pub background: bool,
    pub spill_dir: Option<PathBuf>,
    pub stages: Vec<StageRecord>,
    /// Final job output path when the final stage aliases stored output.
    pub final_output_path: Option<PathBuf>,
    pub final_stdout_inline: Vec<u8>,
    pub final_stderr_inline: Vec<u8>,
}

impl LivePipeline {
    pub(crate) fn new(
        commands: Vec<String>,
        background: bool,
        spill_root: Option<&Path>,
        launch: LaunchId,
    ) -> Self {
        let spill_dir = spill_root.map(|root| root.join(format!("launch-{}", launch.raw())));
        let stages = commands
            .into_iter()
            .enumerate()
            .map(|(step, command)| StageRecord::new(step, command))
            .collect();
        Self {
            background,
            spill_dir,
            stages,
            final_output_path: None,
            final_stdout_inline: Vec::new(),
            final_stderr_inline: Vec::new(),
        }
    }

    pub(crate) fn stage_started(&mut self, step: usize) {
        if let Some(stage) = self.stages.get_mut(step)
            && matches!(stage.status, StageStatus::Pending)
        {
            stage.status = StageStatus::Running;
        }
    }

    pub(crate) async fn append_chunk(
        &mut self,
        step: usize,
        stream: TerminalStream,
        injected_color: bool,
        bytes: Vec<u8>,
    ) {
        let Some(stage) = self.stages.get_mut(step) else {
            return;
        };
        if matches!(stage.status, StageStatus::Pending) {
            stage.status = StageStatus::Running;
        }
        if !stage.retained {
            return;
        }
        let dir = self.spill_dir.as_deref();
        let native = if injected_color {
            strip_sgr_sequences(&bytes)
        } else {
            bytes.clone()
        };
        stage.total_lines += native.split(|&b| b == b'\n').count() - 1;
        stage.ui.append(dir, &format!("s{step}.ui"), &bytes).await;
        stage
            .native_combined
            .append(dir, &format!("s{step}.comb"), &native)
            .await;
        match stream {
            TerminalStream::Stdout => {
                stage
                    .native_stdout
                    .append(dir, &format!("s{step}.out"), &native)
                    .await;
            }
            TerminalStream::Stderr => {
                stage
                    .native_stderr
                    .append(dir, &format!("s{step}.err"), &native)
                    .await;
            }
        }
    }

    pub(crate) async fn stage_exited(
        &mut self,
        step: usize,
        exit_code: i32,
        own_duration: Duration,
        record_after: Duration,
    ) {
        let Some(stage) = self.stages.get_mut(step) else {
            return;
        };
        stage.status = StageStatus::Exited { code: exit_code };
        stage.own_duration = Some(own_duration);
        stage.flush_writers().await;
        if own_duration < record_after {
            stage.retained = false;
            stage.drop_buffers().await;
        } else {
            stage.retained = true;
        }
    }

    /// Finalizes any still-open stages when the overall job finishes or is killed,
    /// and aliases the final stage's native buffers to the job's final output so
    /// no duplicate copy sits on disk.
    pub(crate) async fn settle_on_job_end(
        &mut self,
        record_after: Duration,
        output_path: Option<PathBuf>,
        final_stdout: &[u8],
        final_stderr: &[u8],
    ) {
        for stage in &mut self.stages {
            stage.flush_writers().await;
            if matches!(stage.status, StageStatus::Pending | StageStatus::Running) {
                stage.status = StageStatus::Exited { code: -1 };
                stage.retained = false;
                stage.drop_buffers().await;
            } else if stage.own_duration.is_some_and(|d| d < record_after) {
                stage.retained = false;
                stage.drop_buffers().await;
            }
        }
        let last_idx = self.stages.len().saturating_sub(1);
        if let Some(last) = self.stages.get_mut(last_idx)
            && last.retained
        {
            last.aliased_final = true;
            last.native_combined.clear().await;
            last.native_stdout.clear().await;
            last.native_stderr.clear().await;
            self.final_output_path = output_path;
            if self.final_output_path.is_none() {
                self.final_stdout_inline = final_stdout.to_vec();
            }
            self.final_stderr_inline = final_stderr.to_vec();
        }
    }

    #[must_use]
    pub(crate) fn has_retained_non_final_stage(&self) -> bool {
        let non_final = self.stages.len().saturating_sub(1);
        self.stages[..non_final].iter().any(|s| s.retained)
    }

    #[must_use]
    pub(crate) fn should_retain_after_job(&self) -> bool {
        if self.background {
            self.stages.iter().any(|s| s.retained)
        } else {
            self.has_retained_non_final_stage()
        }
    }

    #[must_use]
    pub(crate) fn kept_steps(&self) -> Vec<usize> {
        self.stages
            .iter()
            .filter(|s| s.retained)
            .map(|s| s.step)
            .collect()
    }

    #[must_use]
    pub(crate) fn active_step(&self, job_running: bool) -> Option<usize> {
        if !job_running {
            return None;
        }
        self.stages
            .iter()
            .find(|s| matches!(s.status, StageStatus::Pending | StageStatus::Running))
            .map(|s| s.step)
    }

    #[must_use]
    pub(crate) fn snapshot(
        &self,
        pid: u32,
        launch: LaunchId,
        job_status: JobStatus,
        started_at: Instant,
        finished_at: Option<Instant>,
        retention: Duration,
    ) -> PipelineSnapshot {
        let running = matches!(job_status, JobStatus::Running);
        let expires_at = finished_at.map(|fin| fin + retention);
        PipelineSnapshot {
            pid,
            launch,
            job_status,
            started_at,
            finished_at,
            expires_at,
            active_step: self.active_step(running),
            stages: self.stages.iter().map(StageRecord::info).collect(),
        }
    }

    pub(crate) async fn read_stage_bytes(
        &self,
        pid: u32,
        step: usize,
        selector: StageStreamSelector,
        raw: bool,
    ) -> Result<Vec<u8>, PipelineReadError> {
        let kept = self.kept_steps();
        let Some(stage) = self.stages.get(step) else {
            return Err(PipelineReadError::InvalidStep {
                pid,
                step,
                kept_steps: kept,
            });
        };
        if !stage.retained {
            return Err(PipelineReadError::InvalidStep {
                pid,
                step,
                kept_steps: kept,
            });
        }

        let bytes = if stage.aliased_final {
            let out_bytes = if let Some(path) = &self.final_output_path {
                async_fs::read(path)
                    .await
                    .map_err(|e| PipelineReadError::Io(e.to_string()))?
            } else {
                self.final_stdout_inline.clone()
            };
            match selector {
                StageStreamSelector::Stdout => out_bytes,
                StageStreamSelector::Stderr => self.final_stderr_inline.clone(),
                StageStreamSelector::Combined => {
                    let mut combined = out_bytes;
                    combined.extend_from_slice(&self.final_stderr_inline);
                    combined
                }
            }
        } else {
            let stream = match selector {
                StageStreamSelector::Combined => &stage.native_combined,
                StageStreamSelector::Stdout => &stage.native_stdout,
                StageStreamSelector::Stderr => &stage.native_stderr,
            };
            stream.read_all().await.map_err(PipelineReadError::Io)?
        };

        if raw {
            Ok(bytes)
        } else {
            Ok(render_terminal_bytes(&bytes))
        }
    }

    pub(crate) async fn read_stage_delta(
        &self,
        pid: u32,
        job_status: JobStatus,
        step: usize,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<JobDelta, PipelineReadError> {
        let full = self
            .read_stage_bytes(pid, step, StageStreamSelector::Combined, false)
            .await?;
        let total_bytes = full.len();
        if cursor >= total_bytes {
            return Ok(JobDelta {
                pid,
                cursor: total_bytes,
                total_bytes,
                bytes: Vec::new(),
                status: job_status,
            });
        }
        let end = cursor.saturating_add(max_bytes).min(total_bytes);
        Ok(JobDelta {
            pid,
            cursor: end,
            total_bytes,
            bytes: full[cursor..end].to_vec(),
            status: job_status,
        })
    }

    pub(crate) async fn read_stage_ui_delta(
        &self,
        pid: u32,
        launch: LaunchId,
        job_status: JobStatus,
        step: usize,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<StageUiDelta, PipelineReadError> {
        let kept = self.kept_steps();
        let Some(stage) = self.stages.get(step) else {
            return Err(PipelineReadError::InvalidStep {
                pid,
                step,
                kept_steps: kept,
            });
        };
        if !stage.retained {
            return Err(PipelineReadError::InvalidStep {
                pid,
                step,
                kept_steps: kept,
            });
        }
        let total_bytes = stage.ui.len();
        let bytes = stage
            .ui
            .read_slice(cursor, max_bytes)
            .await
            .map_err(PipelineReadError::Io)?;
        let next_cursor = cursor.min(total_bytes) + bytes.len();
        Ok(StageUiDelta {
            pid,
            launch,
            step,
            cursor: next_cursor,
            total_bytes,
            bytes,
            stage_status: stage.status,
            job_status,
        })
    }

    pub(crate) async fn destroy(&mut self) {
        for stage in &mut self.stages {
            stage.drop_buffers().await;
        }
        if let Some(dir) = self.spill_dir.take() {
            let _ = async_fs::remove_dir_all(dir).await;
        }
    }
}

/// Completed pipeline retained after its job left the active registry.
#[derive(Debug)]
pub struct RetainedEntry {
    pub pid: u32,
    pub launch: LaunchId,
    pub job_status: JobStatus,
    pub started_at: Instant,
    pub finished_at: Instant,
    pub pipeline: LivePipeline,
}

/// Store of completed retained pipelines and tombstones of expired PIDs.
#[derive(Debug, Default)]
pub struct PipelineRetentionStore {
    pub by_launch: HashMap<LaunchId, RetainedEntry>,
    pub latest_by_pid: HashMap<u32, LaunchId>,
    pub expired_pids: HashSet<u32>,
    /// Stage reads tracked by `(LaunchId, step)` for unchanged-poll refusal.
    pub step_reads: HashMap<(LaunchId, usize), (usize, usize, JobStatus)>,
}

impl PipelineRetentionStore {
    pub(crate) fn on_pid_registered(&mut self, pid: u32) {
        self.latest_by_pid.remove(&pid);
        self.expired_pids.remove(&pid);
    }

    pub(crate) fn insert(&mut self, entry: RetainedEntry) {
        let pid = entry.pid;
        let launch = entry.launch;
        self.expired_pids.remove(&pid);
        self.latest_by_pid.insert(pid, launch);
        self.by_launch.insert(launch, entry);
    }

    #[must_use]
    pub(crate) fn next_expiry(&self, retention: Duration) -> Option<Instant> {
        self.by_launch
            .values()
            .map(|entry| entry.finished_at + retention)
            .min()
    }

    pub(crate) async fn sweep_expired(&mut self, now: Instant, retention: Duration) {
        let expired: Vec<LaunchId> = self
            .by_launch
            .iter()
            .filter(|(_, entry)| entry.finished_at + retention <= now)
            .map(|(launch, _)| *launch)
            .collect();
        for launch in expired {
            if let Some(mut entry) = self.by_launch.remove(&launch) {
                entry.pipeline.destroy().await;
                self.step_reads.retain(|(l, _), _| *l != launch);
                if self.latest_by_pid.get(&entry.pid) == Some(&launch) {
                    self.latest_by_pid.remove(&entry.pid);
                    self.expired_pids.insert(entry.pid);
                }
            }
        }
    }

    pub(crate) fn track_step_delta(
        &mut self,
        launch: LaunchId,
        step: usize,
        delta: JobDelta,
        tracked: bool,
    ) -> JobRead {
        if !tracked {
            return JobRead::Delta(delta);
        }
        let state = (delta.cursor, delta.total_bytes, delta.status.clone());
        let previous = self.step_reads.insert((launch, step), state.clone());
        if delta.bytes.is_empty() && previous.as_ref() == Some(&state) {
            JobRead::Unchanged { pid: delta.pid }
        } else {
            JobRead::Delta(delta)
        }
    }
}
