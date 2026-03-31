use std::io::{self, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use crate::args::{OutputFormat, PermissionMode};
use crate::input::LineEditor;
use crate::render::{Spinner, TerminalRenderer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    pub model: String,
    pub permission_mode: PermissionMode,
    pub config: Option<PathBuf>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionState {
    pub turns: usize,
    pub compacted_messages: usize,
    pub last_model: String,
}

impl SessionState {
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            turns: 0,
            compacted_messages: 0,
            last_model: model.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandResult {
    Continue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Status,
    Compact,
    Unknown(String),
}

impl SlashCommand {
    #[must_use]
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        let command = trimmed
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or_default();
        Some(match command {
            "help" => Self::Help,
            "status" => Self::Status,
            "compact" => Self::Compact,
            other => Self::Unknown(other.to_string()),
        })
    }
}

struct SlashCommandHandler {
    command: SlashCommand,
    summary: &'static str,
}

const SLASH_COMMAND_HANDLERS: &[SlashCommandHandler] = &[
    SlashCommandHandler {
        command: SlashCommand::Help,
        summary: "Show command help",
    },
    SlashCommandHandler {
        command: SlashCommand::Status,
        summary: "Show current session status",
    },
    SlashCommandHandler {
        command: SlashCommand::Compact,
        summary: "Compact local session history",
    },
];

pub struct CliApp {
    config: SessionConfig,
    renderer: TerminalRenderer,
    state: SessionState,
}

impl CliApp {
    #[must_use]
    pub fn new(config: SessionConfig) -> Self {
        let state = SessionState::new(config.model.clone());
        Self {
            config,
            renderer: TerminalRenderer::new(),
            state,
        }
    }

    pub fn run_repl(&mut self) -> io::Result<()> {
        let editor = LineEditor::new("› ");
        println!("Rusty Claude CLI interactive mode");
        println!("Type /help for commands. Shift+Enter or Ctrl+J inserts a newline.");

        while let Some(input) = editor.read_line()? {
            if input.trim().is_empty() {
                continue;
            }

            self.handle_submission(&input, &mut io::stdout())?;
        }

        Ok(())
    }

    pub fn run_prompt(&mut self, prompt: &str, out: &mut impl Write) -> io::Result<()> {
        self.render_response(prompt, out)
    }

    pub fn handle_submission(
        &mut self,
        input: &str,
        out: &mut impl Write,
    ) -> io::Result<CommandResult> {
        if let Some(command) = SlashCommand::parse(input) {
            return self.dispatch_slash_command(command, out);
        }

        self.state.turns += 1;
        self.render_response(input, out)?;
        Ok(CommandResult::Continue)
    }

    fn dispatch_slash_command(
        &mut self,
        command: SlashCommand,
        out: &mut impl Write,
    ) -> io::Result<CommandResult> {
        match command {
            SlashCommand::Help => Self::handle_help(out),
            SlashCommand::Status => self.handle_status(out),
            SlashCommand::Compact => self.handle_compact(out),
            SlashCommand::Unknown(name) => {
                writeln!(out, "Unknown slash command: /{name}")?;
                Ok(CommandResult::Continue)
            }
        }
    }

    fn handle_help(out: &mut impl Write) -> io::Result<CommandResult> {
        writeln!(out, "Available commands:")?;
        for handler in SLASH_COMMAND_HANDLERS {
            let name = match handler.command {
                SlashCommand::Help => "/help",
                SlashCommand::Status => "/status",
                SlashCommand::Compact => "/compact",
                SlashCommand::Unknown(_) => continue,
            };
            writeln!(out, "  {name:<9} {}", handler.summary)?;
        }
        Ok(CommandResult::Continue)
    }

    fn handle_status(&mut self, out: &mut impl Write) -> io::Result<CommandResult> {
        writeln!(
            out,
            "status: turns={} model={} permission-mode={:?} output-format={:?} config={}",
            self.state.turns,
            self.state.last_model,
            self.config.permission_mode,
            self.config.output_format,
            self.config
                .config
                .as_ref()
                .map_or_else(|| String::from("<none>"), |path| path.display().to_string())
        )?;
        Ok(CommandResult::Continue)
    }

    fn handle_compact(&mut self, out: &mut impl Write) -> io::Result<CommandResult> {
        self.state.compacted_messages += self.state.turns;
        self.state.turns = 0;
        writeln!(
            out,
            "Compacted session history into a local summary ({} messages total compacted).",
            self.state.compacted_messages
        )?;
        Ok(CommandResult::Continue)
    }

    fn render_response(&mut self, input: &str, out: &mut impl Write) -> io::Result<()> {
        let mut spinner = Spinner::new();
        for label in [
            "Planning response",
            "Running tool execution",
            "Rendering markdown output",
        ] {
            spinner.tick(label, self.renderer.color_theme(), out)?;
            thread::sleep(Duration::from_millis(24));
        }
        spinner.finish("Streaming response", self.renderer.color_theme(), out)?;

        let response = demo_response(input, &self.config);
        match self.config.output_format {
            OutputFormat::Text => self.renderer.stream_markdown(&response, out)?,
            OutputFormat::Json => writeln!(out, "{{\"message\":{response:?}}}")?,
            OutputFormat::Ndjson => {
                writeln!(out, "{{\"type\":\"message\",\"text\":{response:?}}}")?;
            }
        }
        Ok(())
    }
}

#[must_use]
pub fn demo_response(input: &str, config: &SessionConfig) -> String {
    format!(
        "## Assistant\n\nModel: `{}`  \nPermission mode: `{}`\n\nYou said:\n\n> {}\n\nThis renderer now supports **bold**, *italic*, inline `code`, and syntax-highlighted blocks:\n\n```rust\nfn main() {{\n    println!(\"streaming from rusty-claude-cli\");\n}}\n```",
        config.model,
        permission_mode_label(config.permission_mode),
        input.trim()
    )
}

#[must_use]
pub fn permission_mode_label(mode: PermissionMode) -> &'static str {
    match mode {
        PermissionMode::ReadOnly => "read-only",
        PermissionMode::WorkspaceWrite => "workspace-write",
        PermissionMode::DangerFullAccess => "danger-full-access",
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::args::{OutputFormat, PermissionMode};

    use super::{CliApp, CommandResult, SessionConfig, SlashCommand};

    #[test]
    fn parses_required_slash_commands() {
        assert_eq!(SlashCommand::parse("/help"), Some(SlashCommand::Help));
        assert_eq!(SlashCommand::parse(" /status "), Some(SlashCommand::Status));
        assert_eq!(
            SlashCommand::parse("/compact now"),
            Some(SlashCommand::Compact)
        );
    }

    #[test]
    fn help_status_and_compact_commands_are_wired() {
        let config = SessionConfig {
            model: "claude".into(),
            permission_mode: PermissionMode::WorkspaceWrite,
            config: Some(PathBuf::from("settings.toml")),
            output_format: OutputFormat::Text,
        };
        let mut app = CliApp::new(config);
        let mut out = Vec::new();

        let result = app
            .handle_submission("/help", &mut out)
            .expect("help succeeds");
        assert_eq!(result, CommandResult::Continue);

        app.handle_submission("hello", &mut out)
            .expect("submission succeeds");
        app.handle_submission("/status", &mut out)
            .expect("status succeeds");
        app.handle_submission("/compact", &mut out)
            .expect("compact succeeds");

        let output = String::from_utf8_lossy(&out);
        assert!(output.contains("/help"));
        assert!(output.contains("/status"));
        assert!(output.contains("/compact"));
        assert!(output.contains("status: turns=1"));
        assert!(output.contains("Compacted session history"));
    }
}
