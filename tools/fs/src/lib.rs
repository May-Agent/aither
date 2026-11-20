//! Filesystem abstraction and tool for agent execution.

use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt, fs,
    future::Future,
    io,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
use anyhow::{Result, anyhow};
use arc_swap::ArcSwap;
use askama::Template;
use async_fs::{
    OpenOptions, canonicalize, create_dir_all, read_dir, read_to_string, remove_dir, remove_file,
    symlink_metadata, write,
};
use futures_lite::{AsyncWriteExt, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Abstract filesystem interface for agent tools.
///
/// Implementors can provide virtual or real filesystem access.
pub trait FileSystem: Send + Sync + 'static {
    /// Reads the entire contents of a file as a string.
    fn read_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<String>> + Send + 'a;

    /// Writes the contents to a file, replacing any existing content.
    fn write_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;

    /// Appends the contents to a file, creating it if it does not exist.
    fn append_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;

    /// Removes a file from the filesystem.
    fn remove_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a;

    /// Lists entries in a directory.
    fn list_dir<'a>(
        &'a self,
        dir: &'a Path,
    ) -> impl Future<Output = io::Result<Vec<DirEntry>>> + Send + 'a;

    /// Creates a directory and any necessary parent directories.
    fn create_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a;

    /// Removes a directory and all of its contents.
    fn remove_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a;

    /// Find files matching a glob pattern. Returns relative paths.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the pattern is invalid or scanning fails.
    fn glob<'a>(
        &'a self,
        pattern: &'a str,
    ) -> impl Future<Output = io::Result<Vec<String>>> + Send + 'a {
        // Default implementation returns empty - override for real filesystems
        let _ = pattern;
        std::future::ready(Ok(Vec::new()))
    }
}

/// File system operations: read, write, append, delete, list.
///
/// Provides direct filesystem access within the sandbox. All operations
/// respect the sandbox permission model - writes go to the sandbox directory
/// unless running in unsafe mode.
///
/// Large file reads may be automatically truncated or saved to a separate
/// file. Use standard Unix tools to process large files in parts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum FsOperation {
    /// Read entire file contents.
    Read {
        /// Relative path to the file (e.g., "src/main.rs", "data/config.json").
        path: String,
    },
    /// Write content to a file, creating it if needed, overwriting if exists.
    Write {
        /// Relative path to the file.
        path: String,
        /// Text content to write.
        content: String,
    },
    /// Append content to the end of an existing file.
    Append {
        /// Relative path to the file.
        path: String,
        /// Text content to append.
        content: String,
    },
    /// Delete a file.
    Delete {
        /// Relative path to the file to delete.
        path: String,
    },
    /// List directory contents.
    List {
        /// Relative path to directory. Omit or use empty string for current directory.
        path: Option<String>,
    },
    /// Find files matching a glob pattern recursively.
    /// Examples: "**/*.rs" finds all Rust files, "src/**/*.json" finds JSON in src.
    Glob {
        /// Glob pattern to match (e.g., "**/*.rs", "src/**/*.ts", "*.md").
        pattern: String,
    },
}

/// File system entry information.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DirEntry {
    /// Name of the entry.
    pub name: String,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}

/// A directory listing as the model reads it, rendered by `templates/listing.txt`.
#[derive(Template)]
#[template(path = "listing.txt", escape = "none")]
struct ListingTemplate<'a> {
    entries: &'a [DirEntry],
}

/// Glob matches as the model reads them, rendered by `templates/matches.txt`.
#[derive(Template)]
#[template(path = "matches.txt", escape = "none")]
struct MatchesTemplate<'a> {
    matches: &'a [String],
}

/// Filesystem tool wrapper for agents.
#[derive(Clone, Debug)]
pub struct FileSystemTool<FS> {
    filesystem: FS,
    allow_writes: bool,
    name: String,
}

impl FileSystemTool<LocalFileSystem> {
    /// Create a new filesystem tool rooted at the given path with write access.
    ///
    /// # Panics
    /// Panics if the filesystem initialization fails.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::try_new(root).expect("failed to initialize filesystem tool")
    }

    /// Try to create a new filesystem tool rooted at the given path with write access.
    ///
    /// # Errors
    /// Returns an error if initialization or canonicalization fails.
    pub fn try_new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let fs = LocalFileSystem::new(root)?;
        Ok(Self::with_filesystem(fs))
    }

    /// Create a read-only filesystem tool rooted at the given path.
    ///
    /// # Panics
    /// Panics if initialization fails.
    #[must_use]
    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self::try_read_only(root).expect("failed to initialize filesystem tool")
    }

    /// Try to create a read-only filesystem tool rooted at the given path.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    pub fn try_read_only(root: impl Into<PathBuf>) -> io::Result<Self> {
        let fs = LocalFileSystem::read_only(root)?;
        Ok(Self::with_filesystem(fs).allow_writes(false))
    }
}

impl FileSystemTool<InMemoryFileSystem> {
    /// Create an in-memory filesystem tool instance.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::with_filesystem(InMemoryFileSystem::new())
    }
}

impl<FS: FileSystem> FileSystemTool<FS> {
    /// Create a new `FileSystemTool` wrapping the given filesystem.
    #[must_use]
    pub fn with_filesystem(fs: FS) -> Self {
        Self {
            filesystem: fs,
            allow_writes: true,
            name: "filesystem".into(),
        }
    }

    /// Configure whether writes are allowed.
    #[must_use]
    pub const fn allow_writes(mut self, allow: bool) -> Self {
        self.allow_writes = allow;
        self
    }

    /// Set the tool name.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.allow_writes {
            Ok(())
        } else {
            Err(anyhow!("Filesystem tool is read-only"))
        }
    }
}

impl<FS: FileSystem> Tool for FileSystemTool<FS> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    type Arguments = FsOperation;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        match arguments {
            FsOperation::Read { path } => {
                let content = self
                    .filesystem
                    .read_file(Path::new(&path))
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::text(content))
            }
            FsOperation::Write { path, content } => {
                self.ensure_writable()?;
                self.filesystem
                    .write_file(Path::new(&path), content)
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::Done)
            }
            FsOperation::Append { path, content } => {
                self.ensure_writable()?;
                self.filesystem
                    .append_file(Path::new(&path), content)
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::Done)
            }
            FsOperation::Delete { path } => {
                self.ensure_writable()?;
                self.filesystem
                    .remove_file(Path::new(&path))
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::Done)
            }
            FsOperation::List { path } => {
                let listing = self
                    .filesystem
                    .list_dir(path.as_deref().map_or_else(|| Path::new(""), Path::new))
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::text(
                    ListingTemplate { entries: &listing }.render()?,
                ))
            }
            FsOperation::Glob { pattern } => {
                let matches = self
                    .filesystem
                    .glob(&pattern)
                    .await
                    .map_err(anyhow::Error::new)?;
                Ok(ToolResult::text(
                    MatchesTemplate { matches: &matches }.render()?,
                ))
            }
        }
    }
}

/// Local filesystem implementation backed by standard OS files.
#[derive(Debug, Clone)]
pub struct LocalFileSystem {
    root: PathBuf,
    canonical_root: PathBuf,
    allow_writes: bool,
}

impl LocalFileSystem {
    /// Create a new local filesystem rooted at `root` with write access enabled.
    ///
    /// # Errors
    /// Returns an error if directory creation or canonicalization fails.
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::build(root, true)
    }

    /// Create a new local filesystem rooted at `root` in read-only mode.
    ///
    /// # Errors
    /// Returns an error if directory creation or canonicalization fails.
    pub fn read_only(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::build(root, false)
    }

    fn build(root: impl Into<PathBuf>, allow_writes: bool) -> io::Result<Self> {
        let mut root = root.into();
        if root.as_os_str().is_empty() {
            root = PathBuf::from(".");
        }
        fs::create_dir_all(&root)?;
        let canonical_root = fs::canonicalize(&root)?;
        Ok(Self {
            root: canonical_root.clone(),
            canonical_root,
            allow_writes,
        })
    }

    /// Resolve `relative` to an absolute path inside the root without touching
    /// the filesystem.
    ///
    /// The longest existing prefix (a dangling symlink counts as existing and
    /// fails to canonicalize) is canonicalized, so symlinks and `..` in it are
    /// resolved by the OS; the not-yet-existing remainder may only hold plain
    /// names. The result must lie under the canonical root.
    async fn resolve(&self, relative: &Path) -> io::Result<PathBuf> {
        let candidate = if relative.as_os_str().is_empty() {
            self.root.clone()
        } else {
            self.root.join(relative)
        };
        let components: Vec<Component<'_>> = candidate.components().collect();

        let mut existing = components.len();
        while existing > 0 {
            let prefix: PathBuf = components[..existing].iter().collect();
            if symlink_metadata(&prefix).await.is_ok() {
                break;
            }
            existing -= 1;
        }
        let prefix: PathBuf = components[..existing].iter().collect();
        let mut resolved = canonicalize(&prefix).await?;
        for component in &components[existing..] {
            match component {
                Component::Normal(name) => resolved.push(name),
                Component::CurDir => {}
                _ => return Err(self.escape_error(&candidate)),
            }
        }

        if !resolved.starts_with(&self.canonical_root) {
            return Err(self.escape_error(&resolved));
        }
        Ok(resolved)
    }

    fn escape_error(&self, path: &Path) -> io::Error {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "Access to '{}' escapes filesystem root {}",
                path.display(),
                self.canonical_root.display()
            ),
        )
    }

    async fn create_parent(target: &Path) -> io::Result<()> {
        match target.parent() {
            Some(parent) => create_dir_all(parent).await,
            None => Ok(()),
        }
    }

    fn ensure_writable(&self) -> io::Result<()> {
        if self.allow_writes {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Filesystem is read-only",
            ))
        }
    }
}

impl FileSystem for LocalFileSystem {
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        let target = self.resolve(path).await?;
        read_to_string(&target).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path).await?;
        Self::create_parent(&target).await?;
        write(&target, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path).await?;
        Self::create_parent(&target).await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target)
            .await?;
        file.write_all(contents.as_bytes()).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(path).await?;
        remove_file(&target).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        let target = self.resolve(dir).await?;
        let mut reader = read_dir(&target).await?;
        let mut entries = Vec::new();
        while let Some(entry) = reader.next().await {
            if let Ok(entry) = entry {
                let is_dir = match entry.file_type().await {
                    Ok(ft) if ft.is_dir() => true,
                    Ok(ft) if ft.is_symlink() => async_fs::metadata(entry.path())
                        .await
                        .is_ok_and(|meta| meta.is_dir()),
                    _ => false,
                };
                entries.push(DirEntry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    is_dir,
                });
            }
        }
        Ok(entries)
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(dir).await?;
        create_dir_all(&target).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        self.ensure_writable()?;
        let target = self.resolve(dir).await?;
        remove_dir(&target).await
    }

    async fn glob<'a>(&'a self, pattern: &'a str) -> io::Result<Vec<String>> {
        if Path::new(pattern)
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        {
            return Err(self.escape_error(Path::new(pattern)));
        }
        let root = self.root.clone();
        let canonical_root = self.canonical_root.clone();
        let full_pattern = root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy().into_owned();

        blocking::unblock(move || {
            let entries = glob::glob(&pattern_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.msg))?;

            let mut results = Vec::new();
            for entry in entries.flatten() {
                // Symlinks may lead out of the root: only matches that resolve
                // inside it are reported.
                let Ok(canonical) = fs::canonicalize(&entry) else {
                    continue;
                };
                if !canonical.starts_with(&canonical_root) {
                    continue;
                }
                if let Ok(relative) = entry.strip_prefix(&root) {
                    results.push(relative.to_string_lossy().into_owned());
                }
            }
            results.sort();
            Ok(results)
        })
        .await
    }
}

/// In-memory filesystem implementation.
#[derive(Debug, Clone)]
pub struct InMemoryFileSystem {
    state: Arc<ArcSwap<MemoryState>>,
}

#[derive(Debug, Default, Clone)]
struct MemoryState {
    entries: BTreeMap<PathBuf, MemoryEntry>,
}

#[derive(Debug, Clone)]
enum MemoryEntry {
    File(String),
    Directory,
}

impl InMemoryFileSystem {
    /// Create a new in-memory filesystem instance.
    #[must_use]
    pub fn new() -> Self {
        let mut entries = BTreeMap::new();
        entries.insert(PathBuf::from("/"), MemoryEntry::Directory);
        Self {
            state: Arc::new(ArcSwap::from_pointee(MemoryState { entries })),
        }
    }

    fn mutate_entries(
        &self,
        mut update: impl FnMut(&mut BTreeMap<PathBuf, MemoryEntry>) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut outcome = Ok(());
        self.state.rcu(|current| {
            let mut entries = current.entries.clone();
            match update(&mut entries) {
                Ok(()) => {
                    outcome = Ok(());
                    Arc::new(MemoryState { entries })
                }
                Err(error) => {
                    outcome = Err(error);
                    Arc::clone(current)
                }
            }
        });
        outcome
    }

    fn normalize(path: &Path) -> io::Result<PathBuf> {
        let mut normalized = PathBuf::from("/");
        for component in path.components() {
            match component {
                Component::Normal(part) => normalized.push(part),
                Component::ParentDir => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "Parent directory traversal is not allowed",
                    ));
                }
                Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
            }
        }
        Ok(normalized)
    }

    fn ensure_dir_present(entries: &mut BTreeMap<PathBuf, MemoryEntry>, dir: &Path) {
        for ancestor in dir.ancestors() {
            if ancestor.as_os_str().is_empty() {
                continue;
            }
            entries
                .entry(ancestor.to_path_buf())
                .or_insert(MemoryEntry::Directory);
        }
    }

    fn list_children(entries: &BTreeMap<PathBuf, MemoryEntry>, dir: &Path) -> Vec<DirEntry> {
        let mut result = Vec::new();
        for (path, entry) in entries {
            if let Some(parent) = path.parent()
                && parent == dir
            {
                let name = path
                    .file_name()
                    .map_or_else(|| "/".into(), |n| n.to_string_lossy().into_owned());
                let is_dir = matches!(entry, MemoryEntry::Directory);
                if name != "/" {
                    result.push(DirEntry { name, is_dir });
                }
            }
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

impl Default for InMemoryFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystem for InMemoryFileSystem {
    fn read_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<String>> + Send + 'a {
        let result = Self::normalize(path).and_then(|path| {
            let state = self.state.load();
            match state.entries.get(&path) {
                Some(MemoryEntry::File(contents)) => Ok(contents.clone()),
                Some(MemoryEntry::Directory) => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Cannot read a directory",
                )),
                None => Err(io::Error::new(io::ErrorKind::NotFound, "File not found")),
            }
        });
        std::future::ready(result)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        let result = Self::normalize(path).and_then(|path| {
            self.mutate_entries(|entries| {
                if let Some(parent) = path.parent() {
                    Self::ensure_dir_present(entries, parent);
                }
                entries.insert(path.clone(), MemoryEntry::File(contents.clone()));
                Ok(())
            })
        });
        std::future::ready(result)
    }

    fn append_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        let result = Self::normalize(path).and_then(|path| {
            self.mutate_entries(|entries| {
                if let Some(entry) = entries.get_mut(&path) {
                    match entry {
                        MemoryEntry::File(existing) => existing.push_str(&contents),
                        MemoryEntry::Directory => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "Cannot append to a directory",
                            ));
                        }
                    }
                } else {
                    if let Some(parent) = path.parent() {
                        Self::ensure_dir_present(entries, parent);
                    }
                    entries.insert(path.clone(), MemoryEntry::File(contents.clone()));
                }
                Ok(())
            })
        });
        std::future::ready(result)
    }

    fn remove_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        let result = Self::normalize(path).and_then(|path| {
            self.mutate_entries(|entries| match entries.remove(&path) {
                Some(MemoryEntry::File(_)) => Ok(()),
                Some(MemoryEntry::Directory) => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Use remove_dir for directories",
                )),
                None => Err(io::Error::new(io::ErrorKind::NotFound, "File not found")),
            })
        });
        std::future::ready(result)
    }

    fn list_dir<'a>(
        &'a self,
        dir: &'a Path,
    ) -> impl Future<Output = io::Result<Vec<DirEntry>>> + Send + 'a {
        let result = Self::normalize(dir).and_then(|dir| {
            let state = self.state.load();
            if !matches!(state.entries.get(&dir), Some(MemoryEntry::Directory)) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "Directory does not exist",
                ));
            }
            Ok(Self::list_children(&state.entries, &dir))
        });
        std::future::ready(result)
    }

    fn create_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        let result = Self::normalize(dir).and_then(|dir| {
            self.mutate_entries(|entries| {
                if let Some(parent) = dir.parent() {
                    Self::ensure_dir_present(entries, parent);
                }
                entries.insert(dir.clone(), MemoryEntry::Directory);
                Ok(())
            })
        });
        std::future::ready(result)
    }

    fn remove_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        let result = Self::normalize(dir).and_then(|dir| {
            if dir == Path::new("/") {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Cannot remove root directory",
                ));
            }
            self.mutate_entries(|entries| {
                let mut to_remove = Vec::new();
                for path in entries.keys() {
                    if path.starts_with(&dir) {
                        to_remove.push(path.clone());
                    }
                }
                if to_remove.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        "Directory does not exist",
                    ));
                }
                for path in to_remove {
                    entries.remove(&path);
                }
                Ok(())
            })
        });
        std::future::ready(result)
    }
}

/// Mount-table based composite filesystem.
#[derive(Clone)]
pub struct MountFileSystem<FS> {
    mounts: Arc<ArcSwap<Vec<MountPoint<FS>>>>,
}

struct MountPoint<FS> {
    prefix: PathBuf,
    fs: Arc<FS>,
}

impl<FS> Clone for MountPoint<FS> {
    fn clone(&self) -> Self {
        Self {
            prefix: self.prefix.clone(),
            fs: Arc::clone(&self.fs),
        }
    }
}

impl<FS> fmt::Debug for MountPoint<FS>
where
    FS: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MountPoint")
            .field("prefix", &self.prefix)
            .field("fs", &self.fs)
            .finish()
    }
}

impl<FS> MountFileSystem<FS>
where
    FS: FileSystem,
{
    /// Creates a new, empty mount-table filesystem.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mounts: Arc::new(ArcSwap::from_pointee(Vec::new())),
        }
    }

    /// Mounts a filesystem at the given path prefix.
    #[must_use]
    pub fn mount(self, prefix: impl Into<PathBuf>, fs: FS) -> Self {
        let prefix = normalize_mount(&prefix.into());
        let fs = Arc::new(fs);
        self.mounts.rcu(|current| {
            let mut mounts = current.as_ref().clone();
            mounts.push(MountPoint {
                prefix: prefix.clone(),
                fs: Arc::clone(&fs),
            });
            Arc::new(mounts)
        });
        self
    }

    fn route(&self, path: &Path) -> io::Result<(Arc<FS>, PathBuf)> {
        let normalized = normalize_mount(path);
        let mounts = self.mounts.load();
        let mut best: Option<&MountPoint<_>> = None;
        for mount in mounts.iter() {
            if normalized.starts_with(&mount.prefix) {
                match best {
                    Some(current)
                        if current.prefix.components().count()
                            >= mount.prefix.components().count() => {}
                    _ => best = Some(mount),
                }
            }
        }
        let mount =
            best.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No mount for path"))?;
        let relative = normalized
            .strip_prefix(&mount.prefix)
            .unwrap_or_else(|_| Path::new(""))
            .to_path_buf();
        Ok((Arc::clone(&mount.fs), relative))
    }
}

impl<FS> Default for MountFileSystem<FS>
where
    FS: FileSystem,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<FS> fmt::Debug for MountFileSystem<FS>
where
    FS: FileSystem,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mounts = self.mounts.load();
        let prefixes: Vec<PathBuf> = mounts.iter().map(|mount| mount.prefix.clone()).collect();
        f.debug_struct("MountFileSystem")
            .field("mounts", &prefixes)
            .finish()
    }
}

impl<FS> FileSystem for MountFileSystem<FS>
where
    FS: FileSystem,
{
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().read_file(&relative).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().write_file(&relative, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().append_file(&relative, contents).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(path)?;
        fs.as_ref().remove_file(&relative).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().list_dir(&relative).await
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().create_dir(&relative).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        let (fs, relative) = self.route(dir)?;
        fs.as_ref().remove_dir(&relative).await
    }
}

fn normalize_mount(path: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        return PathBuf::from("/");
    }
    let mut normalized = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    normalized
}

/// File permissions wrapper struct.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
pub struct FilePermissions {
    /// Read permission.
    pub read: bool,
    /// Write permission.
    pub write: bool,
    /// List permission.
    pub list: bool,
    /// Delete permission.
    pub delete: bool,
}

impl FilePermissions {
    /// Create read-only permissions.
    #[must_use]
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            list: true,
            delete: false,
        }
    }

    /// Create read-write permissions.
    #[must_use]
    pub const fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            list: true,
            delete: true,
        }
    }
}

/// Permissioned filesystem wrapper enforcing access control rules.
#[derive(Debug, Clone)]
pub struct PermissionedFileSystem<FS> {
    inner: FS,
    permissions: FilePermissions,
}

impl<FS> PermissionedFileSystem<FS> {
    /// Create a new permissioned filesystem wrapper.
    #[must_use]
    pub const fn new(inner: FS, permissions: FilePermissions) -> Self {
        Self { inner, permissions }
    }

    fn require(allowed: bool, action: &str) -> io::Result<()> {
        if allowed {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Operation '{action}' is not permitted"),
            ))
        }
    }
}

impl<FS> FileSystem for PermissionedFileSystem<FS>
where
    FS: FileSystem,
{
    async fn read_file<'a>(&'a self, path: &'a Path) -> io::Result<String> {
        Self::require(self.permissions.read, "read")?;
        self.inner.read_file(path).await
    }

    async fn write_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        Self::require(self.permissions.write, "write")?;
        self.inner.write_file(path, contents).await
    }

    async fn append_file<'a>(&'a self, path: &'a Path, contents: String) -> io::Result<()> {
        Self::require(self.permissions.write, "append")?;
        self.inner.append_file(path, contents).await
    }

    async fn remove_file<'a>(&'a self, path: &'a Path) -> io::Result<()> {
        Self::require(self.permissions.delete, "remove_file")?;
        self.inner.remove_file(path).await
    }

    async fn list_dir<'a>(&'a self, dir: &'a Path) -> io::Result<Vec<DirEntry>> {
        Self::require(self.permissions.list, "list_dir")?;
        self.inner.list_dir(dir).await
    }

    async fn create_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        Self::require(self.permissions.write, "create_dir")?;
        self.inner.create_dir(dir).await
    }

    async fn remove_dir<'a>(&'a self, dir: &'a Path) -> io::Result<()> {
        Self::require(self.permissions.delete, "remove_dir")?;
        self.inner.remove_dir(dir).await
    }
}

/// Filesystem hook event.
#[derive(Debug, Clone)]
pub struct FsHook {
    /// The triggered operation.
    pub operation: FsHookOperation,
}

/// Operations intercepted by the filesystem hook.
#[derive(Debug, Clone)]
pub enum FsHookOperation {
    /// Read operation.
    Read {
        /// Path of the file.
        path: PathBuf,
    },
    /// Write operation.
    Write {
        /// Path of the file.
        path: PathBuf,
        /// Number of bytes written.
        bytes: usize,
    },
    /// Append operation.
    Append {
        /// Path of the file.
        path: PathBuf,
        /// Number of bytes appended.
        bytes: usize,
    },
    /// Remove file operation.
    RemoveFile {
        /// Path of the file.
        path: PathBuf,
    },
    /// List directory operation.
    List {
        /// Path of the directory.
        path: PathBuf,
    },
    /// Create directory operation.
    CreateDir {
        /// Path of the directory.
        path: PathBuf,
    },
    /// Remove directory operation.
    RemoveDir {
        /// Path of the directory.
        path: PathBuf,
    },
}

/// Filesystem wrapper that emits hooks on operations.
#[derive(Clone)]
pub struct HookedFileSystem<FS> {
    inner: FS,
    hook: Arc<dyn Fn(FsHook) + Send + Sync>,
}

impl<FS> fmt::Debug for HookedFileSystem<FS>
where
    FS: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookedFileSystem")
            .field("inner", &self.inner)
            .field("hook", &"<closure>")
            .finish()
    }
}

impl<FS> HookedFileSystem<FS> {
    /// Create a new hooked filesystem wrapper.
    #[must_use]
    pub fn new(inner: FS, hook: impl Fn(FsHook) + Send + Sync + 'static) -> Self {
        Self {
            inner,
            hook: Arc::new(hook),
        }
    }

    fn emit(&self, operation: FsHookOperation) {
        (self.hook)(FsHook { operation });
    }
}

impl<FS> FileSystem for HookedFileSystem<FS>
where
    FS: FileSystem,
{
    fn read_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<String>> + Send + 'a {
        self.emit(FsHookOperation::Read {
            path: path.to_path_buf(),
        });
        self.inner.read_file(path)
    }

    fn write_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::Write {
            path: path.to_path_buf(),
            bytes: contents.len(),
        });
        self.inner.write_file(path, contents)
    }

    fn append_file<'a>(
        &'a self,
        path: &'a Path,
        contents: String,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::Append {
            path: path.to_path_buf(),
            bytes: contents.len(),
        });
        self.inner.append_file(path, contents)
    }

    fn remove_file<'a>(
        &'a self,
        path: &'a Path,
    ) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::RemoveFile {
            path: path.to_path_buf(),
        });
        self.inner.remove_file(path)
    }

    fn list_dir<'a>(
        &'a self,
        dir: &'a Path,
    ) -> impl Future<Output = io::Result<Vec<DirEntry>>> + Send + 'a {
        self.emit(FsHookOperation::List {
            path: dir.to_path_buf(),
        });
        self.inner.list_dir(dir)
    }

    fn create_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::CreateDir {
            path: dir.to_path_buf(),
        });
        self.inner.create_dir(dir)
    }

    fn remove_dir<'a>(&'a self, dir: &'a Path) -> impl Future<Output = io::Result<()>> + Send + 'a {
        self.emit(FsHookOperation::RemoveDir {
            path: dir.to_path_buf(),
        });
        self.inner.remove_dir(dir)
    }
}

#[cfg(test)]
mod escape_tests;

#[cfg(test)]
mod template_tests {
    use super::*;

    #[test]
    fn listing_marks_directories() {
        let entries = [
            DirEntry {
                name: "src".into(),
                is_dir: true,
            },
            DirEntry {
                name: "Cargo.toml".into(),
                is_dir: false,
            },
        ];
        assert_eq!(
            ListingTemplate { entries: &entries }.render().unwrap(),
            "src/\nCargo.toml\n"
        );
        assert_eq!(
            ListingTemplate { entries: &[] }.render().unwrap(),
            "The directory is empty."
        );
    }

    #[test]
    fn matches_are_one_per_line() {
        let matches = ["a.rs".to_string(), "b/c.rs".to_string()];
        assert_eq!(
            MatchesTemplate { matches: &matches }.render().unwrap(),
            "a.rs\nb/c.rs\n"
        );
        assert_eq!(
            MatchesTemplate { matches: &[] }.render().unwrap(),
            "No files match."
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn in_memory_fs_preserves_concurrent_writes_and_ancestor_dirs() {
        let fs = InMemoryFileSystem::new();
        let mut tasks = Vec::new();
        for index in 0..32_usize {
            let fs_clone = fs.clone();
            tasks.push(tokio::spawn(async move {
                let path = PathBuf::from(format!("nested/group_{}/item_{index}.txt", index % 4));
                fs_clone
                    .write_file(&path, format!("payload-{index}"))
                    .await
                    .unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        for index in 0..32_usize {
            let path = PathBuf::from(format!("nested/group_{}/item_{index}.txt", index % 4));
            assert_eq!(
                fs.read_file(&path).await.unwrap(),
                format!("payload-{index}")
            );
        }
        let root_entries = fs.list_dir(Path::new("")).await.unwrap();
        assert!(root_entries.iter().any(|e| e.name == "nested" && e.is_dir));
        let nested_entries = fs.list_dir(Path::new("nested")).await.unwrap();
        assert_eq!(nested_entries.len(), 4);
    }
}
