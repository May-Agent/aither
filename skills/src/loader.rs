//! Skill loader for loading skills from the filesystem.

use std::path::{Component, Path, PathBuf};

use async_fs as afs;
use futures_lite::stream::StreamExt;

use crate::{Skill, SkillError};

/// Loads skills from filesystem directories.
///
/// # Example
///
/// ```rust,ignore
/// let loader = SkillLoader::new()
///     .add_path("~/.aither/skills")
///     .add_path("./.aither/skills");
///
/// let skills = loader.load_all().await?;
/// ```
#[derive(Debug, Default)]
pub struct SkillLoader {
    paths: Vec<PathBuf>,
}

impl SkillLoader {
    /// Create a new skill loader with no paths.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path to search for skills.
    ///
    /// Each path should be a directory containing skill subdirectories.
    /// The structure expected is:
    /// ```text
    /// path/
    /// └── skill-name/
    ///     └── SKILL.md
    /// ```
    #[must_use]
    pub fn add_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.paths.push(path.into());
        self
    }

    /// Load all skills from all configured paths.
    ///
    /// Skills are loaded from subdirectories of each path.
    /// Each subdirectory must contain a `SKILL.md` file.
    ///
    /// # Errors
    ///
    /// Returns an error if a skill directory cannot be read or parsed.
    pub async fn load_all(&self) -> Result<Vec<Skill>, SkillError> {
        let mut skills = Vec::new();

        for base_path in &self.paths {
            if !path_exists_async(base_path).await? {
                continue;
            }

            let mut entries =
                afs::read_dir(base_path)
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: base_path.clone(),
                        source,
                    })?;

            while let Some(entry) =
                entries
                    .try_next()
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: base_path.clone(),
                        source,
                    })?
            {
                let path = entry.path();
                let is_dir = afs::metadata(&path)
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: path.clone(),
                        source,
                    })?
                    .is_dir();

                if is_dir {
                    let skill = Self::load_from_dir(&path).await?;
                    skills.push(skill);
                }
            }
        }

        Ok(skills)
    }

    /// Load a specific skill by name.
    ///
    /// Searches all configured paths for a skill with the given name.
    ///
    /// # Errors
    ///
    /// Returns `SkillError::InvalidStructure` if `name` is not a single normal
    /// path component, `SkillError::NotFound` if no skill with the given name
    /// exists, or other errors if the skill cannot be parsed.
    pub async fn load(&self, name: &str) -> Result<Skill, SkillError> {
        if !is_single_normal_component(name) {
            return Err(SkillError::InvalidStructure {
                path: PathBuf::from(name),
                reason: "skill name must be a single normal path component".to_string(),
            });
        }

        for base_path in &self.paths {
            let skill_path = base_path.join(name);
            if path_exists_async(&skill_path).await? {
                return Self::load_from_dir(&skill_path).await;
            }
        }

        Err(SkillError::NotFound {
            name: name.to_string(),
        })
    }

    /// Load a skill from a specific directory.
    async fn load_from_dir(dir: &Path) -> Result<Skill, SkillError> {
        let skill_file = dir.join("SKILL.md");

        if !path_exists_async(&skill_file).await? {
            return Err(SkillError::InvalidStructure {
                path: dir.to_path_buf(),
                reason: "missing SKILL.md file".to_string(),
            });
        }

        let content =
            afs::read_to_string(&skill_file)
                .await
                .map_err(|source| SkillError::ReadFile {
                    path: skill_file,
                    source,
                })?;

        let mut skill = Skill::parse(&content)?;

        // Load additional resources from the directory
        Self::load_resources(dir, &mut skill).await?;

        Ok(skill)
    }

    /// Load additional resources (templates, scripts) from the skill directory.
    async fn load_resources(dir: &Path, skill: &mut Skill) -> Result<(), SkillError> {
        // Load templates
        let templates_dir = dir.join("templates");
        if path_exists_async(&templates_dir).await? {
            Self::load_dir_contents(&templates_dir, "templates/", skill).await?;
        }

        // Load scripts
        let scripts_dir = dir.join("scripts");
        if path_exists_async(&scripts_dir).await? {
            Self::load_dir_contents(&scripts_dir, "scripts/", skill).await?;
        }

        Ok(())
    }

    /// Load all files from a directory into the skill's resources.
    async fn load_dir_contents(
        dir: &Path,
        prefix: &str,
        skill: &mut Skill,
    ) -> Result<(), SkillError> {
        let mut entries = afs::read_dir(dir)
            .await
            .map_err(|source| SkillError::ReadFile {
                path: dir.to_path_buf(),
                source,
            })?;

        while let Some(entry) = entries
            .try_next()
            .await
            .map_err(|source| SkillError::ReadFile {
                path: dir.to_path_buf(),
                source,
            })?
        {
            let path = entry.path();
            let is_file = afs::metadata(&path)
                .await
                .map_err(|source| SkillError::ReadFile {
                    path: path.clone(),
                    source,
                })?
                .is_file();

            if !is_file {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let bytes = afs::read(&path)
                    .await
                    .map_err(|source| SkillError::ReadFile {
                        path: path.clone(),
                        source,
                    })?;
                let Ok(content) = String::from_utf8(bytes) else {
                    continue;
                };
                let key = format!("{prefix}{name}");
                skill.resources.insert(key, content);
            }
        }

        Ok(())
    }
}

fn is_single_normal_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    ) && !name.contains('\\')
}

async fn path_exists_async(path: &Path) -> Result<bool, SkillError> {
    match afs::metadata(path).await {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(SkillError::ReadFile {
            path: path.to_path_buf(),
            source: err,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn minimal_skill_content(name: &str, description: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n\nInstructions here.\n")
    }

    fn create_skill_dir(base: &Path, name: &str) -> PathBuf {
        let skill_dir = base.join(name);
        fs::create_dir(&skill_dir).expect("create skill directory");
        skill_dir
    }

    fn create_skill_with_content(base: &Path, name: &str, content: &str) -> PathBuf {
        let skill_dir = create_skill_dir(base, name);
        fs::write(skill_dir.join("SKILL.md"), content).expect("write SKILL.md");
        skill_dir
    }

    fn prepare_loader() -> (TempDir, SkillLoader) {
        let dir = tempdir().expect("create temp dir");
        let loader = SkillLoader::new().add_path(dir.path());
        (dir, loader)
    }

    #[tokio::test]
    async fn test_load_skill_from_dir() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "test-skill",
            &minimal_skill_content("test-skill", "A test skill"),
        );

        let skills = loader.load_all().await.expect("load skill");

        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "test-skill");
    }

    #[tokio::test]
    async fn test_load_skill_by_name() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "my-skill",
            &minimal_skill_content("my-skill", "My skill"),
        );

        let skill = loader.load("my-skill").await.expect("load skill by name");

        assert_eq!(skill.name, "my-skill");
    }

    #[tokio::test]
    async fn test_load_skill_not_found() {
        let (_dir, loader) = prepare_loader();
        let result = loader.load("nonexistent").await;

        assert!(matches!(result, Err(SkillError::NotFound { .. })));
    }

    #[tokio::test]
    async fn test_load_resources() {
        let (dir, loader) = prepare_loader();
        let skill_dir = create_skill_with_content(
            dir.path(),
            "with-resources",
            &minimal_skill_content("with-resources", "Skill with resources"),
        );

        // Create templates directory
        let templates_dir = skill_dir.join("templates");
        fs::create_dir(&templates_dir).expect("create templates dir");
        fs::write(templates_dir.join("review.md"), "# Review Template").expect("write template");

        let skill = loader
            .load("with-resources")
            .await
            .expect("load skill with resources");

        assert_eq!(
            skill.resources.get("templates/review.md"),
            Some(&"# Review Template".to_string())
        );
    }

    #[tokio::test]
    async fn test_load_all_fails_on_missing_skill_file() {
        let (dir, loader) = prepare_loader();
        create_skill_dir(dir.path(), "missing-markdown");

        let result = loader.load_all().await;
        assert!(matches!(result, Err(SkillError::InvalidStructure { .. })));
    }

    #[tokio::test]
    async fn test_load_all_fails_on_malformed_frontmatter() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(dir.path(), "broken", "name: broken");

        let result = loader.load_all().await;
        assert!(matches!(result, Err(SkillError::MissingFrontmatter)));
    }

    #[tokio::test]
    async fn test_load_all_fails_fast_when_any_skill_is_invalid() {
        let (dir, loader) = prepare_loader();
        create_skill_with_content(
            dir.path(),
            "valid-skill",
            &minimal_skill_content("valid-skill", "A valid skill"),
        );
        create_skill_dir(dir.path(), "invalid-skill");

        let result = loader.load_all().await;
        assert!(matches!(result, Err(SkillError::InvalidStructure { .. })));
    }

    #[tokio::test]
    async fn test_load_rejects_path_traversal_and_non_single_names() {
        let (dir, loader) = prepare_loader();
        let outside = tempdir().expect("outside temp dir");
        create_skill_with_content(
            outside.path(),
            "secret-skill",
            &minimal_skill_content("secret-skill", "Outside skill"),
        );

        let relative = format!(
            "../{}",
            outside.path().file_name().unwrap().to_str().unwrap()
        );
        for bad_name in [
            "",
            ".",
            "..",
            "../secret-skill",
            "a/b",
            "a\\b",
            "/etc/passwd",
            &relative,
            outside.path().join("secret-skill").to_str().unwrap(),
        ] {
            let result = loader.load(bad_name).await;
            assert!(
                matches!(result, Err(SkillError::InvalidStructure { .. })),
                "expected InvalidStructure for {bad_name:?}, got {result:?}"
            );
        }
        drop(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_load_all_follows_symlinked_skill_directories() {
        let target_root = tempdir().expect("target dir");
        let target_skill = create_skill_with_content(
            target_root.path(),
            "linked-skill",
            &minimal_skill_content("linked-skill", "Symlinked skill"),
        );

        let (dir, loader) = prepare_loader();
        std::os::unix::fs::symlink(&target_skill, dir.path().join("linked-skill"))
            .expect("create symlink");

        let skills = loader.load_all().await.expect("load symlinked skill");
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "linked-skill");
    }

    #[tokio::test]
    async fn test_load_tolerates_non_utf8_files_in_scripts_and_templates() {
        let (dir, loader) = prepare_loader();
        let skill_dir = create_skill_with_content(
            dir.path(),
            "binary-scripts",
            "---\nname: binary-scripts\ndescription: Skill with binary\nlicense: MIT\n---\n\nBody.\n",
        );

        let scripts_dir = skill_dir.join("scripts");
        fs::create_dir(&scripts_dir).expect("create scripts dir");
        fs::write(scripts_dir.join("tool.bin"), [0xff, 0xfe, 0x00, 0x80]).expect("write bin");
        fs::write(scripts_dir.join("run.sh"), "#!/bin/sh\necho hi\n").expect("write script");

        let templates_dir = skill_dir.join("templates");
        fs::create_dir(&templates_dir).expect("create templates dir");
        fs::write(
            templates_dir.join("icon.bin"),
            [0x89, 0x50, 0x4e, 0x47, 0xff],
        )
        .expect("write");

        let skills = loader
            .load_all()
            .await
            .expect("load all with binary scripts");
        assert_eq!(skills.len(), 1);
        assert_eq!(
            skills[0]
                .resources
                .get("scripts/run.sh")
                .map(String::as_str),
            Some("#!/bin/sh\necho hi\n")
        );
        assert!(!skills[0].resources.contains_key("scripts/tool.bin"));
        assert!(!skills[0].resources.contains_key("templates/icon.bin"));
    }
}
