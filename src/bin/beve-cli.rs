use std::path::{Path, PathBuf};
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        print_usage();
        process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let result = match args[1].as_str() {
        "to-json" => cmd_to_json(&args[2..]),
        "to-mat" => cmd_to_mat(&args[2..]),
        "from-json" => cmd_from_json(&args[2..]),
        other => {
            eprintln!("error: unknown command: {other}");
            eprintln!();
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn print_usage() {
    eprintln!("Usage: beve-cli <command> [options] <input> [output]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  to-json    Convert BEVE to JSON");
    eprintln!("  to-mat     Convert BEVE to MATLAB v7.3 MAT");
    eprintln!("  from-json  Convert JSON to BEVE");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  beve-cli to-json data.beve              # writes data.json");
    eprintln!("  beve-cli to-mat data.beve               # writes data.mat");
    eprintln!("  beve-cli to-mat data.beve output.mat    # explicit output path");
    eprintln!("  beve-cli from-json data.json             # writes data.beve");
    eprintln!();
    eprintln!("Run `beve-cli <command> --help` for command-specific options.");
}

// ---------------------------------------------------------------------------
// to-json
// ---------------------------------------------------------------------------

fn cmd_to_json(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: beve to-json <input.beve> [output.json]");
        eprintln!();
        eprintln!("If no output path is given, replaces the .beve extension with .json.");
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let (input, positional) = parse_positional(args)?;

    let output = positional_or_default_ext(&positional, &input, "json");

    let beve_bytes = std::fs::read(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let json_bytes = beve::beve_slice_to_json(&beve_bytes).map_err(|e| e.to_string())?;
    std::fs::write(&output, &json_bytes).map_err(|e| format!("{}: {e}", output.display()))?;

    println!("{}", output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// from-json
// ---------------------------------------------------------------------------

fn cmd_from_json(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: beve from-json <input.json> [output.beve]");
        eprintln!();
        eprintln!("If no output path is given, replaces the .json extension with .beve.");
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let (input, positional) = parse_positional(args)?;

    let output = positional_or_default_ext(&positional, &input, "beve");

    let json_bytes = std::fs::read(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let beve_bytes = beve::json_slice_to_beve(&json_bytes).map_err(|e| e.to_string())?;
    std::fs::write(&output, &beve_bytes).map_err(|e| format!("{}: {e}", output.display()))?;

    println!("{}", output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// to-mat
// ---------------------------------------------------------------------------

fn cmd_to_mat(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: beve to-mat [options] <input.beve> [output.mat]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --name <var>   Store as a single variable (default: \"data\")");
        eprintln!("  --workspace    Expand top-level object keys into workspace variables");
        eprintln!();
        eprintln!("If no output path is given, replaces the .beve extension with .mat.");
        process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let mut workspace = false;
    let mut var_name = String::from("data");
    let mut positional = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workspace" => workspace = true,
            "--name" => {
                i += 1;
                if i >= args.len() {
                    return Err("--name requires a value".into());
                }
                var_name = args[i].clone();
            }
            arg if arg.starts_with('-') => return Err(format!("unknown option: {arg}")),
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }

    if positional.is_empty() {
        return Err("no input file specified".into());
    }

    let input = PathBuf::from(&positional[0]);
    if !input.exists() {
        return Err(format!("file not found: {}", input.display()));
    }

    let output = if positional.len() >= 2 {
        PathBuf::from(&positional[1])
    } else {
        input.with_extension("mat")
    };

    let root = if workspace {
        beve::RootBinding::WorkspaceObject
    } else {
        beve::RootBinding::NamedVariable(&var_name)
    };
    let options = beve::MatV73Options {
        invalid_name_policy: beve::InvalidNamePolicy::Sanitize,
        ..beve::MatV73Options::default()
    };

    beve::beve_file_to_mat_v73_file(&input, &output, root, &options).map_err(|e| e.to_string())?;

    println!("{}", output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_positional(args: &[String]) -> Result<(PathBuf, Vec<String>), String> {
    let mut positional = Vec::new();
    for arg in args {
        if !arg.starts_with('-') {
            positional.push(arg.clone());
        }
    }
    if positional.is_empty() {
        return Err("no input file specified".into());
    }
    let input = PathBuf::from(&positional[0]);
    if !input.exists() {
        return Err(format!("file not found: {}", input.display()));
    }
    Ok((input, positional))
}

fn positional_or_default_ext(positional: &[String], input: &Path, ext: &str) -> PathBuf {
    if positional.len() >= 2 {
        PathBuf::from(&positional[1])
    } else {
        input.with_extension(ext)
    }
}
