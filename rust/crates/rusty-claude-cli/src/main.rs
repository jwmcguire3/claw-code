mod app;
mod args;
mod input;
mod render;

use std::path::PathBuf;

use app::{CliApp, SessionConfig};
use args::{Cli, Command};
use clap::Parser;
use compat_harness::{extract_manifest, UpstreamPaths};
use runtime::BootstrapPlan;

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Some(Command::DumpManifests) => dump_manifests(),
        Some(Command::BootstrapPlan) => {
            print_bootstrap_plan();
            Ok(())
        }
        Some(Command::Prompt { prompt }) => {
            let joined = prompt.join(" ");
            let mut app = CliApp::new(build_session_config(&cli));
            app.run_prompt(&joined, &mut std::io::stdout())
        }
        None => {
            let mut app = CliApp::new(build_session_config(&cli));
            app.run_repl()
        }
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn build_session_config(cli: &Cli) -> SessionConfig {
    SessionConfig {
        model: cli.model.clone(),
        permission_mode: cli.permission_mode,
        config: cli.config.clone(),
        output_format: cli.output_format,
    }
}

fn dump_manifests() -> std::io::Result<()> {
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = UpstreamPaths::from_workspace_dir(&workspace_dir);
    let manifest = extract_manifest(&paths)?;
    println!("commands: {}", manifest.commands.entries().len());
    println!("tools: {}", manifest.tools.entries().len());
    println!("bootstrap phases: {}", manifest.bootstrap.phases().len());
    Ok(())
}

fn print_bootstrap_plan() {
    for phase in BootstrapPlan::claude_code_default().phases() {
        println!("- {phase:?}");
    }
}
