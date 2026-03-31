use regex::RegexBuilder;
use serde::Serialize;
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolManifestEntry {
    pub name: String,
    pub source: ToolSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSource {
    Base,
    Conditional,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolRegistry {
    entries: Vec<ToolManifestEntry>,
}

impl ToolRegistry {
    #[must_use]
    pub fn new(entries: Vec<ToolManifestEntry>) -> Self {
        Self { entries }
    }

    #[must_use]
    pub fn entries(&self) -> &[ToolManifestEntry] {
        &self.entries
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TextContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolResult {
    pub content: Vec<TextContent>,
}

impl ToolResult {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![TextContent {
                kind: "text",
                text: text.into(),
            }],
        }
    }
}

#[derive(Debug)]
pub struct ToolError {
    message: Cow<'static, str>,
}

impl ToolError {
    #[must_use]
    pub fn new(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<io::Error> for ToolError {
    fn from(value: io::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<regex::Error> for ToolError {
    fn from(value: regex::Error) -> Self {
        Self::new(value.to_string())
    }
}

pub trait Tool {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> Value;
    fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;
}

fn schema_string(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

fn schema_number(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

fn schema_boolean(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

fn strict_object(properties: &Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn parse_string(input: &Value, key: &'static str) -> Result<String, ToolError> {
    input
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| ToolError::new(format!("missing or invalid string field: {key}")))
}

fn optional_string(input: &Value, key: &'static str) -> Result<Option<String>, ToolError> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(ToolError::new(format!("invalid string field: {key}"))),
    }
}

fn optional_u64(input: &Value, key: &'static str) -> Result<Option<u64>, ToolError> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| ToolError::new(format!("invalid numeric field: {key}")))
            .map(Some),
    }
}

fn optional_bool(input: &Value, key: &'static str) -> Result<Option<bool>, ToolError> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| ToolError::new(format!("invalid boolean field: {key}")))
            .map(Some),
    }
}

fn absolute_path(path: &str) -> Result<PathBuf, ToolError> {
    let expanded = if let Some(rest) = path.strip_prefix("~/") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map_or_else(|| PathBuf::from(path), |home| home.join(rest))
    } else {
        PathBuf::from(path)
    };

    if expanded.is_absolute() {
        Ok(expanded)
    } else {
        Err(ToolError::new(format!("path must be absolute: {path}")))
    }
}

fn relative_display(path: &Path, base: &Path) -> String {
    path.strip_prefix(base).ok().map_or_else(
        || path.to_string_lossy().replace('\\', "/"),
        |value| value.to_string_lossy().replace('\\', "/"),
    )
}

fn line_slice(content: &str, offset: Option<u64>, limit: Option<u64>) -> String {
    let start = usize_from_u64(offset.unwrap_or(1).saturating_sub(1));
    let lines: Vec<&str> = content.lines().collect();
    let end = limit
        .map_or(lines.len(), |limit| {
            start.saturating_add(usize_from_u64(limit))
        })
        .min(lines.len());

    if start >= lines.len() {
        return String::new();
    }

    lines[start..end]
        .iter()
        .enumerate()
        .map(|(index, line)| format!("{:>6}\t{line}", start + index + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_page_range(pages: &str) -> Result<(u64, u64), ToolError> {
    if let Some((start, end)) = pages.split_once('-') {
        let start = start
            .trim()
            .parse::<u64>()
            .map_err(|_| ToolError::new("invalid pages parameter"))?;
        let end = end
            .trim()
            .parse::<u64>()
            .map_err(|_| ToolError::new("invalid pages parameter"))?;
        if start == 0 || end < start {
            return Err(ToolError::new("invalid pages parameter"));
        }
        Ok((start, end))
    } else {
        let page = pages
            .trim()
            .parse::<u64>()
            .map_err(|_| ToolError::new("invalid pages parameter"))?;
        if page == 0 {
            return Err(ToolError::new("invalid pages parameter"));
        }
        Ok((page, page))
    }
}

fn apply_single_edit(
    original: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<String, ToolError> {
    if old_string == new_string {
        return Err(ToolError::new(
            "No changes to make: old_string and new_string are exactly the same.",
        ));
    }

    if old_string.is_empty() {
        if original.is_empty() {
            return Ok(new_string.to_owned());
        }
        return Err(ToolError::new(
            "Cannot create new file - file already exists.",
        ));
    }

    let matches = original.matches(old_string).count();
    if matches == 0 {
        return Err(ToolError::new(format!(
            "String to replace not found in file.\nString: {old_string}"
        )));
    }

    if matches > 1 && !replace_all {
        return Err(ToolError::new(format!(
            "Found {matches} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.\nString: {old_string}"
        )));
    }

    let updated = if replace_all {
        original.replace(old_string, new_string)
    } else {
        original.replacen(old_string, new_string, 1)
    };
    Ok(updated)
}

fn diff_hunks(_before: &str, _after: &str) -> Value {
    json!([])
}

fn usize_from_u64(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

pub struct BashTool;
pub struct ReadTool;
pub struct WriteTool;
pub struct EditTool;
pub struct GlobTool;
pub struct GrepTool;

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "Bash"
    }

    fn description(&self) -> &'static str {
        "Execute a shell command in the current environment."
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "command": schema_string("The command to execute"),
                "timeout": schema_number("Optional timeout in milliseconds (max 600000)"),
                "description": schema_string("Clear, concise description of what this command does in active voice. Never use words like \"complex\" or \"risk\" in the description - just describe what it does."),
                "run_in_background": schema_boolean("Set to true to run this command in the background. Use Read to read the output later."),
                "dangerouslyDisableSandbox": schema_boolean("Set this to true to dangerously override sandbox mode and run commands without sandboxing.")
            }),
            &["command"],
        )
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let command = parse_string(&input, "command")?;
        let _timeout = optional_u64(&input, "timeout")?;
        let _description = optional_string(&input, "description")?;
        let run_in_background = optional_bool(&input, "run_in_background")?.unwrap_or(false);
        let _disable_sandbox = optional_bool(&input, "dangerouslyDisableSandbox")?.unwrap_or(false);

        if run_in_background {
            return Ok(ToolResult::text(
                "Background execution is not supported in this runtime.",
            ));
        }

        let output = Command::new("bash").arg("-lc").arg(&command).output()?;
        let mut rendered = String::new();
        if !output.stdout.is_empty() {
            rendered.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !rendered.is_empty() && !rendered.ends_with('\n') {
                rendered.push('\n');
            }
            rendered.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        if rendered.is_empty() {
            rendered = if output.status.success() {
                "Done".to_owned()
            } else {
                format!("Command exited with status {}", output.status)
            };
        }
        Ok(ToolResult::text(rendered.trim_end().to_owned()))
    }
}

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "Read"
    }

    fn description(&self) -> &'static str {
        "Read a file from the local filesystem."
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "file_path": schema_string("The absolute path to the file to read"),
                "offset": json!({"type":"number","description":"The line number to start reading from. Only provide if the file is too large to read at once","minimum":0}),
                "limit": json!({"type":"number","description":"The number of lines to read. Only provide if the file is too large to read at once.","exclusiveMinimum":0}),
                "pages": schema_string("Page range for PDF files (e.g., \"1-5\", \"3\", \"10-20\"). Only applicable to PDF files. Maximum 20 pages per request.")
            }),
            &["file_path"],
        )
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let file_path = parse_string(&input, "file_path")?;
        let path = absolute_path(&file_path)?;
        let offset = optional_u64(&input, "offset")?;
        let limit = optional_u64(&input, "limit")?;
        let pages = optional_string(&input, "pages")?;

        let content = fs::read_to_string(&path)?;
        if path.extension().and_then(|ext| ext.to_str()) == Some("pdf") {
            if let Some(pages) = pages {
                let (start, end) = parse_page_range(&pages)?;
                return Ok(ToolResult::text(format!(
                    "PDF page extraction is not implemented in Rust yet for {}. Requested pages {}-{}.",
                    path.display(), start, end
                )));
            }
        }

        let rendered = if offset.is_some() || limit.is_some() {
            line_slice(&content, offset, limit)
        } else {
            line_slice(&content, Some(1), None)
        };
        Ok(ToolResult::text(rendered))
    }
}

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "Write"
    }

    fn description(&self) -> &'static str {
        "Write a file to the local filesystem."
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "file_path": schema_string("The absolute path to the file to write (must be absolute, not relative)"),
                "content": schema_string("The content to write to the file")
            }),
            &["file_path", "content"],
        )
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let file_path = parse_string(&input, "file_path")?;
        let content = parse_string(&input, "content")?;
        let path = absolute_path(&file_path)?;
        let existed = path.exists();
        let original = if existed {
            Some(fs::read_to_string(&path)?)
        } else {
            None
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &content)?;

        let payload = json!({
            "type": if existed { "update" } else { "create" },
            "filePath": file_path,
            "content": content,
            "structuredPatch": diff_hunks(original.as_deref().unwrap_or(""), &content),
            "originalFile": original,
            "gitDiff": Value::Null,
        });
        Ok(ToolResult::text(payload.to_string()))
    }
}

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "Edit"
    }

    fn description(&self) -> &'static str {
        "A tool for editing files"
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "file_path": schema_string("The absolute path to the file to modify"),
                "old_string": schema_string("The text to replace"),
                "new_string": schema_string("The text to replace it with (must be different from old_string)"),
                "replace_all": json!({"type":"boolean","description":"Replace all occurrences of old_string (default false)","default":false})
            }),
            &["file_path", "old_string", "new_string"],
        )
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let file_path = parse_string(&input, "file_path")?;
        let old_string = parse_string(&input, "old_string")?;
        let new_string = parse_string(&input, "new_string")?;
        let replace_all = optional_bool(&input, "replace_all")?.unwrap_or(false);
        let path = absolute_path(&file_path)?;
        let original = if path.exists() {
            fs::read_to_string(&path)?
        } else {
            String::new()
        };
        let updated = apply_single_edit(&original, &old_string, &new_string, replace_all)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &updated)?;

        let payload = json!({
            "filePath": file_path,
            "oldString": old_string,
            "newString": new_string,
            "originalFile": original,
            "structuredPatch": diff_hunks("", ""),
            "userModified": false,
            "replaceAll": replace_all,
            "gitDiff": Value::Null,
        });
        Ok(ToolResult::text(payload.to_string()))
    }
}

impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "Glob"
    }

    fn description(&self) -> &'static str {
        "Fast file pattern matching tool"
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "pattern": schema_string("The glob pattern to match files against"),
                "path": schema_string("The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter \"undefined\" or \"null\" - simply omit it for the default behavior. Must be a valid directory path if provided.")
            }),
            &["pattern"],
        )
    }

    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let pattern = parse_string(&input, "pattern")?;
        let root = optional_string(&input, "path")?
            .map(|path| absolute_path(&path))
            .transpose()?
            .unwrap_or(std::env::current_dir()?);
        let start = std::time::Instant::now();
        let mut filenames = Vec::new();
        visit_files(&root, &mut |path| {
            let relative = relative_display(path, &root);
            if glob_matches(&pattern, &relative) {
                filenames.push(relative);
            }
        })?;
        filenames.sort();
        let truncated = filenames.len() > 100;
        if truncated {
            filenames.truncate(100);
        }
        let payload = json!({
            "durationMs": start.elapsed().as_millis(),
            "numFiles": filenames.len(),
            "filenames": filenames,
            "truncated": truncated,
        });
        Ok(ToolResult::text(payload.to_string()))
    }
}

impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "Grep"
    }

    fn description(&self) -> &'static str {
        "Fast content search tool"
    }

    fn input_schema(&self) -> Value {
        strict_object(
            &json!({
                "pattern": schema_string("The regular expression pattern to search for in file contents"),
                "path": schema_string("File or directory to search in (rg PATH). Defaults to current working directory."),
                "glob": schema_string("Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob"),
                "output_mode": {"type":"string","enum":["content","files_with_matches","count"],"description":"Output mode: \"content\" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), \"files_with_matches\" shows file paths (supports head_limit), \"count\" shows match counts (supports head_limit). Defaults to \"files_with_matches\"."},
                "-B": schema_number("Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."),
                "-A": schema_number("Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."),
                "-C": schema_number("Alias for context."),
                "context": schema_number("Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."),
                "-n": {"type":"boolean","description":"Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."},
                "-i": schema_boolean("Case insensitive search (rg -i)"),
                "type": schema_string("File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."),
                "head_limit": schema_number("Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly — large result sets waste context)."),
                "offset": schema_number("Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."),
                "multiline": schema_boolean("Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.")
            }),
            &["pattern"],
        )
    }

    #[allow(clippy::too_many_lines)]
    fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let pattern = parse_string(&input, "pattern")?;
        let root = optional_string(&input, "path")?
            .map(|path| absolute_path(&path))
            .transpose()?
            .unwrap_or(std::env::current_dir()?);
        let glob = optional_string(&input, "glob")?;
        let output_mode = optional_string(&input, "output_mode")?
            .unwrap_or_else(|| "files_with_matches".to_owned());
        let context_before = usize_from_u64(optional_u64(&input, "-B")?.unwrap_or(0));
        let context_after = usize_from_u64(optional_u64(&input, "-A")?.unwrap_or(0));
        let context_c = optional_u64(&input, "-C")?;
        let context = optional_u64(&input, "context")?;
        let show_line_numbers = optional_bool(&input, "-n")?.unwrap_or(true);
        let case_insensitive = optional_bool(&input, "-i")?.unwrap_or(false);
        let file_type = optional_string(&input, "type")?;
        let head_limit = optional_u64(&input, "head_limit")?;
        let offset = usize_from_u64(optional_u64(&input, "offset")?.unwrap_or(0));
        let _multiline = optional_bool(&input, "multiline")?.unwrap_or(false);

        let shared_context = usize_from_u64(context.or(context_c).unwrap_or(0));
        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(case_insensitive)
            .build()?;

        let mut matched_lines = Vec::new();
        let mut files_with_matches = Vec::new();
        let mut count_lines = Vec::new();
        let mut total_matches = 0usize;

        let candidates = collect_files(&root)?;
        for path in candidates {
            let relative = relative_display(&path, &root);
            if !matches_file_filter(&relative, glob.as_deref(), file_type.as_deref()) {
                continue;
            }
            let Ok(file_content) = fs::read_to_string(&path) else {
                continue;
            };
            let lines: Vec<&str> = file_content.lines().collect();
            let mut matched_indexes = Vec::new();
            let mut file_match_count = 0usize;
            for (index, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    matched_indexes.push(index);
                    file_match_count += regex.find_iter(line).count().max(1);
                }
            }
            if matched_indexes.is_empty() {
                continue;
            }
            total_matches += file_match_count;
            files_with_matches.push(relative.clone());
            count_lines.push(format!("{relative}:{file_match_count}"));

            if output_mode == "content" {
                let mut included = BTreeSet::new();
                for index in matched_indexes {
                    let before = if shared_context > 0 {
                        shared_context
                    } else {
                        context_before
                    };
                    let after = if shared_context > 0 {
                        shared_context
                    } else {
                        context_after
                    };
                    let start = index.saturating_sub(before);
                    let end = (index + after).min(lines.len().saturating_sub(1));
                    for line_index in start..=end {
                        included.insert(line_index);
                    }
                }
                for line_index in included {
                    if show_line_numbers {
                        matched_lines.push(format!(
                            "{relative}:{}:{}",
                            line_index + 1,
                            lines[line_index]
                        ));
                    } else {
                        matched_lines.push(format!("{relative}:{}", lines[line_index]));
                    }
                }
            }
        }

        let rendered = match output_mode.as_str() {
            "content" => {
                let limited = apply_offset_limit(matched_lines, head_limit, offset);
                json!({
                    "mode": "content",
                    "numFiles": 0,
                    "filenames": [],
                    "content": limited.join("\n"),
                    "numLines": limited.len(),
                    "appliedOffset": (offset > 0).then_some(offset),
                })
            }
            "count" => {
                let limited = apply_offset_limit(count_lines, head_limit, offset);
                json!({
                    "mode": "count",
                    "numFiles": files_with_matches.len(),
                    "filenames": [],
                    "content": limited.join("\n"),
                    "numMatches": total_matches,
                    "appliedOffset": (offset > 0).then_some(offset),
                })
            }
            _ => {
                files_with_matches.sort();
                let limited = apply_offset_limit(files_with_matches, head_limit, offset);
                json!({
                    "mode": "files_with_matches",
                    "numFiles": limited.len(),
                    "filenames": limited,
                    "appliedOffset": (offset > 0).then_some(offset),
                })
            }
        };

        Ok(ToolResult::text(rendered.to_string()))
    }
}

fn apply_offset_limit<T>(items: Vec<T>, limit: Option<u64>, offset: usize) -> Vec<T> {
    let mut iter = items.into_iter().skip(offset);
    match limit {
        Some(0) | None => iter.collect(),
        Some(limit) => iter.by_ref().take(usize_from_u64(limit)).collect(),
    }
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>, ToolError> {
    let mut files = Vec::new();
    if root.is_file() {
        files.push(root.to_path_buf());
        return Ok(files);
    }
    visit_files(root, &mut |path| files.push(path.to_path_buf()))?;
    Ok(files)
}

fn visit_files(root: &Path, visitor: &mut dyn FnMut(&Path)) -> Result<(), ToolError> {
    if root.is_file() {
        visitor(root);
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_files(&path, visitor)?;
        } else if path.is_file() {
            visitor(&path);
        }
    }
    Ok(())
}

fn matches_file_filter(relative: &str, glob: Option<&str>, file_type: Option<&str>) -> bool {
    let glob_ok = glob.is_none_or(|pattern| {
        split_glob_patterns(pattern)
            .into_iter()
            .any(|single| glob_matches(&single, relative))
    });
    let type_ok = file_type.is_none_or(|kind| path_matches_type(relative, kind));
    glob_ok && type_ok
}

fn split_glob_patterns(patterns: &str) -> Vec<String> {
    let mut result = Vec::new();
    for raw in patterns.split_whitespace() {
        if raw.contains('{') && raw.contains('}') {
            result.push(raw.to_owned());
        } else {
            result.extend(
                raw.split(',')
                    .filter(|part| !part.is_empty())
                    .map(ToOwned::to_owned),
            );
        }
    }
    result
}

fn path_matches_type(relative: &str, kind: &str) -> bool {
    let extension = Path::new(relative)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    matches!(
        (kind, extension),
        ("rust", "rs")
            | ("js", "js")
            | ("ts", "ts")
            | ("tsx", "tsx")
            | ("py", "py")
            | ("go", "go")
            | ("java", "java")
            | ("json", "json")
            | ("md", "md")
    )
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    expand_braces(pattern)
        .into_iter()
        .any(|expanded| glob_match_one(&expanded, path))
}

fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(start) = pattern.find('{') else {
        return vec![pattern.to_owned()];
    };
    let Some(end_rel) = pattern[start..].find('}') else {
        return vec![pattern.to_owned()];
    };
    let end = start + end_rel;
    let prefix = &pattern[..start];
    let suffix = &pattern[end + 1..];
    pattern[start + 1..end]
        .split(',')
        .flat_map(|middle| expand_braces(&format!("{prefix}{middle}{suffix}")))
        .collect()
}

fn glob_match_one(pattern: &str, path: &str) -> bool {
    let pattern = pattern.replace('\\', "/");
    let path = path.replace('\\', "/");
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    glob_match_parts(&pattern_parts, &path_parts)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }
    if pattern[0] == "**" {
        if glob_match_parts(&pattern[1..], path) {
            return true;
        }
        if !path.is_empty() {
            return glob_match_parts(pattern, &path[1..]);
        }
        return false;
    }
    if path.is_empty() {
        return false;
    }
    if segment_matches(pattern[0], path[0]) {
        return glob_match_parts(&pattern[1..], &path[1..]);
    }
    false
}

fn segment_matches(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti, mut star_idx, mut match_idx) = (0usize, 0usize, None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_idx = Some(pi);
            match_idx = ti;
            pi += 1;
        } else if let Some(star) = star_idx {
            pi = star + 1;
            match_idx += 1;
            ti = match_idx;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

#[must_use]
pub fn core_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(BashTool),
        Box::new(ReadTool),
        Box::new(WriteTool),
        Box::new(EditTool),
        Box::new(GlobTool),
        Box::new(GrepTool),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn text(result: &ToolResult) -> String {
        result.content[0].text.clone()
    }

    #[test]
    fn manifests_core_tools() {
        let names: Vec<_> = core_tools().into_iter().map(|tool| tool.name()).collect();
        assert_eq!(names, vec!["Bash", "Read", "Write", "Edit", "Glob", "Grep"]);
    }

    #[test]
    fn bash_executes_command() {
        let result = BashTool
            .execute(json!({ "command": "printf 'hello'" }))
            .unwrap();
        assert_eq!(text(&result), "hello");
    }

    #[test]
    fn read_schema_matches_expected_keys() {
        let schema = ReadTool.input_schema();
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(schema["required"], json!(["file_path"]));
        assert!(properties.contains_key("file_path"));
        assert!(properties.contains_key("offset"));
        assert!(properties.contains_key("limit"));
        assert!(properties.contains_key("pages"));
    }

    #[test]
    fn read_returns_numbered_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();

        let result = ReadTool
            .execute(json!({ "file_path": path.to_string_lossy(), "offset": 2, "limit": 1 }))
            .unwrap();

        assert_eq!(text(&result), "     2\tbeta");
    }

    #[test]
    fn write_creates_file_and_reports_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("new.txt");
        let result = WriteTool
            .execute(json!({ "file_path": path.to_string_lossy(), "content": "hello" }))
            .unwrap();
        let payload: Value = serde_json::from_str(&text(&result)).unwrap();
        assert_eq!(payload["type"], "create");
        assert_eq!(fs::read_to_string(path).unwrap(), "hello");
    }

    #[test]
    fn edit_replaces_single_match() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("edit.txt");
        fs::write(&path, "hello world\n").unwrap();
        let result = EditTool
            .execute(json!({
                "file_path": path.to_string_lossy(),
                "old_string": "world",
                "new_string": "rust",
                "replace_all": false
            }))
            .unwrap();
        let payload: Value = serde_json::from_str(&text(&result)).unwrap();
        assert_eq!(payload["replaceAll"], false);
        assert_eq!(fs::read_to_string(path).unwrap(), "hello rust\n");
    }

    #[test]
    fn glob_finds_matching_files() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        fs::write(dir.path().join("src/nested/main.rs"), "").unwrap();
        fs::write(dir.path().join("README.md"), "").unwrap();

        let result = GlobTool
            .execute(json!({ "pattern": "**/*.rs", "path": dir.path().to_string_lossy() }))
            .unwrap();
        let payload: Value = serde_json::from_str(&text(&result)).unwrap();
        assert_eq!(payload["numFiles"], 2);
        assert_eq!(
            payload["filenames"],
            json!(["src/lib.rs", "src/nested/main.rs"])
        );
    }

    #[test]
    fn grep_supports_file_list_mode() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "fn main() {}\nlet alpha = 1;\n").unwrap();
        fs::write(dir.path().join("b.txt"), "alpha\nalpha\n").unwrap();

        let result = GrepTool
            .execute(json!({
                "pattern": "alpha",
                "path": dir.path().to_string_lossy(),
                "output_mode": "files_with_matches"
            }))
            .unwrap();
        let payload: Value = serde_json::from_str(&text(&result)).unwrap();
        assert_eq!(payload["filenames"], json!(["a.rs", "b.txt"]));
    }

    #[test]
    fn grep_supports_content_and_count_modes() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("a.rs"), "alpha\nbeta\nalpha\n").unwrap();

        let content = GrepTool
            .execute(json!({
                "pattern": "alpha",
                "path": dir.path().to_string_lossy(),
                "output_mode": "content",
                "-n": true
            }))
            .unwrap();
        let content_payload: Value = serde_json::from_str(&text(&content)).unwrap();
        assert_eq!(content_payload["numLines"], 2);
        assert!(content_payload["content"]
            .as_str()
            .unwrap()
            .contains("a.rs:1:alpha"));

        let count = GrepTool
            .execute(json!({
                "pattern": "alpha",
                "path": dir.path().to_string_lossy(),
                "output_mode": "count"
            }))
            .unwrap();
        let count_payload: Value = serde_json::from_str(&text(&count)).unwrap();
        assert_eq!(count_payload["numMatches"], 2);
        assert_eq!(count_payload["content"], "a.rs:2");
    }
}
