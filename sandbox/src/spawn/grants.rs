//! Runtime-mutable filesystem grants of a [`Sandbox`](crate::Sandbox).
//!
//! A [`Grants`] handle is cheap to clone and shares one store with every
//! clone: a host keeps the handle, adds grants when a workspace request is
//! approved, and every process spawned afterwards carries them.
//!
//! A spawn snapshots the grants once, when it launches: a process already
//! running keeps the grants it started with, so a change applies to the
//! next spawn — see [`Grants::add_readable`].

use std::path::PathBuf;
use std::sync::Arc;

use arc_swap::ArcSwap;

/// The path grants one spawn launches with. Immutable once handed out:
/// snapshots are replaced, never edited.
#[derive(Debug, Clone, Default)]
pub struct GrantSnapshot {
    /// Paths a sandboxed process may read but not write.
    pub readable: Vec<PathBuf>,
    /// Paths a sandboxed process may write to (and therefore also read).
    pub writable: Vec<PathBuf>,
    /// Absolute paths a sandboxed process may not read, on top of heel's
    /// credential table. Everything else on the host stays readable.
    pub read_denied: Vec<PathBuf>,
    /// The access to the common git directory of a writable linked worktree.
    pub git_dir: GitDirAccess,
}

/// The access a sandbox has to the common git directory of a linked
/// worktree it may write.
///
/// That directory is the main repository's, so only the session's own
/// sandbox writes it; a child's sandbox reads it, and commits only through
/// a commit grant's repository of its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GitDirAccess {
    /// Read, write and execute: the session's own sandbox.
    #[default]
    ReadWrite,
    /// Read only: a child's sandbox.
    ReadOnly,
}

impl GitDirAccess {
    /// The access to the common git directory of a worktree the sandbox
    /// can write.
    #[must_use]
    pub fn writable_worktree_access(self) -> heel::Access {
        match self {
            Self::ReadWrite => heel::Access::WRITE | heel::Access::EXEC,
            Self::ReadOnly => heel::Access::READ,
        }
    }
}

#[derive(Debug, Clone)]
enum GrantsInner {
    /// The sandbox's own store.
    Own(Arc<ArcSwap<GrantSnapshot>>),
    /// A child's view: its parent's readable side plus its own grants.
    /// Readable grants added to the parent later reach the child; writable
    /// grants added to the parent never become writable here, and nothing
    /// the child adds reaches the parent.
    Derived {
        parent: Box<Grants>,
        grants: Arc<ArcSwap<GrantSnapshot>>,
    },
}

/// Shared, runtime-mutable grants of a [`Sandbox`](crate::Sandbox).
///
/// Cloning the handle keeps the same underlying store: every clone,
/// including the clones a sandbox makes for its own bookkeeping and for
/// children (see [`Self::derived`]), sees grants added through any handle.
#[derive(Debug, Clone)]
pub struct Grants {
    inner: GrantsInner,
}

impl Default for Grants {
    fn default() -> Self {
        Self::new()
    }
}

impl Grants {
    /// An empty grant store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: GrantsInner::Own(Arc::new(ArcSwap::from_pointee(GrantSnapshot::default()))),
        }
    }

    /// A child's view of these grants: the snapshot merges this handle's
    /// readable and read-denied lists (kept live) with `own`'s, and takes
    /// only `own`'s writable list. Writable grants of this handle are never
    /// writable through the view, the view's git directory access is
    /// read-only, and nothing granted through the view reaches this handle.
    #[must_use]
    pub fn derived(&self, own: GrantSnapshot) -> Self {
        Self {
            inner: GrantsInner::Derived {
                parent: Box::new(self.clone()),
                grants: Arc::new(ArcSwap::from_pointee(GrantSnapshot {
                    git_dir: GitDirAccess::ReadOnly,
                    ..own
                })),
            },
        }
    }

    /// Whether this handle is a child's view of another's.
    #[must_use]
    pub const fn is_derived(&self) -> bool {
        matches!(self.inner, GrantsInner::Derived { .. })
    }

    const fn own_store(&self) -> &Arc<ArcSwap<GrantSnapshot>> {
        match &self.inner {
            GrantsInner::Own(store) | GrantsInner::Derived { grants: store, .. } => store,
        }
    }

    fn update_own(&self, mutate: impl Fn(&mut GrantSnapshot)) {
        self.own_store().rcu(|current| {
            let mut next = (**current).clone();
            mutate(&mut next);
            next
        });
    }

    /// Adds `path` as a readable grant. Processes spawned after this call
    /// read it; processes already running keep the grants they launched
    /// with. Dedupes; a path already writable stays writable and is not
    /// also listed readable.
    pub fn add_readable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            if !set.writable.contains(&path) && !set.readable.contains(&path) {
                set.readable.push(path.clone());
            }
        });
    }

    /// Adds `path` as a writable grant, with the same "from the next spawn
    /// on" semantics as [`Self::add_readable`]. A writable grant subsumes a
    /// readable one for the same path.
    pub fn add_writable(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            set.readable.retain(|entry| *entry != path);
            if !set.writable.contains(&path) {
                set.writable.push(path.clone());
            }
        });
    }

    /// Denies sandboxed processes reading `path` and everything beneath it,
    /// beyond heel's credential table, whatever else is granted. The path
    /// must be absolute. Same "from the next spawn on" semantics as
    /// [`Self::add_readable`]; a child's view also denies what its parent
    /// denies.
    pub fn deny_read(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            if !set.read_denied.contains(&path) {
                set.read_denied.push(path.clone());
            }
        });
    }

    /// Takes back the readable and the writable grant of `path`, with the
    /// same "from the next spawn on" semantics as [`Self::add_readable`]:
    /// processes already running keep the grants they launched with. Nothing
    /// of the path is left readable by this call; read access stays only
    /// through a readable grant added again. On a child's view this removes
    /// only its own grants, never one its parent holds.
    pub fn revoke(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.update_own(|set| {
            set.readable.retain(|entry| *entry != path);
            set.writable.retain(|entry| *entry != path);
        });
    }

    /// One consistent view of the grants, taken at spawn time.
    ///
    /// For a derived handle this merges the parent's current readable and
    /// read-denied lists with this handle's own, so readable grants added to
    /// the parent between the derivation and this snapshot are included.
    #[must_use]
    pub fn snapshot(&self) -> Arc<GrantSnapshot> {
        match &self.inner {
            GrantsInner::Own(store) => store.load_full(),
            GrantsInner::Derived {
                parent,
                grants: own,
            } => {
                let parent_set = parent.snapshot();
                let own_set = own.load_full();
                Arc::new(GrantSnapshot {
                    readable: parent_set
                        .readable
                        .iter()
                        .cloned()
                        .chain(own_set.readable.iter().cloned())
                        .collect(),
                    writable: own_set.writable.clone(),
                    read_denied: parent_set
                        .read_denied
                        .iter()
                        .cloned()
                        .chain(own_set.read_denied.iter().cloned())
                        .collect(),
                    git_dir: GitDirAccess::ReadOnly,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revoking_a_grant_takes_it_back_from_the_next_snapshot() {
        let grants = Grants::new();
        grants.add_readable("/data/readable");
        grants.add_writable("/data/writable");
        let launched = grants.snapshot();

        grants.revoke("/data/readable");
        grants.revoke("/data/writable");

        let next = grants.snapshot();
        assert!(next.readable.is_empty(), "{next:?}");
        assert!(next.writable.is_empty(), "{next:?}");
        assert_eq!(launched.readable, [PathBuf::from("/data/readable")]);
        assert_eq!(launched.writable, [PathBuf::from("/data/writable")]);
    }

    #[test]
    fn a_grant_added_later_reaches_the_next_snapshot_only() {
        let grants = Grants::new();
        let before = grants.snapshot();
        grants.add_writable("/data/work");
        grants.deny_read("/data/secret");
        let after = grants.snapshot();
        assert!(before.writable.is_empty() && before.read_denied.is_empty());
        assert_eq!(after.writable, [PathBuf::from("/data/work")]);
        assert_eq!(after.read_denied, [PathBuf::from("/data/secret")]);
    }

    #[test]
    fn a_writable_grant_subsumes_a_readable_one() {
        let grants = Grants::new();
        grants.add_readable("/data/shared");
        grants.add_writable("/data/shared");
        grants.add_readable("/data/shared");
        let snapshot = grants.snapshot();
        assert!(snapshot.readable.is_empty(), "{snapshot:?}");
        assert_eq!(snapshot.writable, [PathBuf::from("/data/shared")]);
    }

    #[test]
    fn a_child_removes_only_its_own_grants() {
        let parent = Grants::new();
        parent.add_readable("/data/parent");
        let child = parent.derived(GrantSnapshot {
            readable: vec!["/data/child".into()],
            ..GrantSnapshot::default()
        });

        child.revoke("/data/parent");
        child.revoke("/data/child");

        assert_eq!(child.snapshot().readable, [PathBuf::from("/data/parent")]);
        assert_eq!(parent.snapshot().readable, [PathBuf::from("/data/parent")]);
    }

    #[test]
    fn a_derived_view_follows_its_parent_readable_side_but_never_widens_it() {
        let parent = Grants::new();
        parent.add_writable("/data/parent-work");
        parent.deny_read("/data/parent-secret");
        let child = parent.derived(GrantSnapshot {
            writable: vec!["/data/child-work".into()],
            ..GrantSnapshot::default()
        });
        assert!(child.is_derived() && !parent.is_derived());

        // A readable grant the parent adds later reaches the child.
        parent.add_readable("/data/later");
        let view = child.snapshot();
        assert_eq!(view.readable, [PathBuf::from("/data/later")]);
        // The parent's writable grants are never the child's; its denies are.
        assert_eq!(view.writable, [PathBuf::from("/data/child-work")]);
        assert_eq!(view.read_denied, [PathBuf::from("/data/parent-secret")]);
        assert_eq!(view.git_dir, GitDirAccess::ReadOnly);

        // Nothing the child grants reaches the parent.
        child.add_writable("/data/child-more");
        child.add_readable("/data/child-read");
        child.deny_read("/data/child-secret");
        let parent_view = parent.snapshot();
        assert_eq!(parent_view.writable, [PathBuf::from("/data/parent-work")]);
        assert_eq!(parent_view.readable, [PathBuf::from("/data/later")]);
        assert_eq!(
            parent_view.read_denied,
            [PathBuf::from("/data/parent-secret")]
        );
        assert_eq!(parent_view.git_dir, GitDirAccess::ReadWrite);
    }
}
