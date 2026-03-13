use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Default)]
struct NativeLinkConfig {
    include_paths: Vec<PathBuf>,
    link_paths: Vec<PathBuf>,
    libs: Vec<String>,
    frameworks: Vec<String>,
    framework_paths: Vec<PathBuf>,
    raw_link_args: Vec<String>,
}

fn main() {
    println!("cargo:rerun-if-changed=tests/cpp/glaze_interop.cpp");
    println!("cargo:rerun-if-changed=tests/cpp/matio_oracle.cpp");
    println!("cargo:rerun-if-changed=reference/glaze/include");

    for var in [
        "CXX",
        "MATIO_INCLUDE_DIR",
        "MATIO_INCLUDE_DIRS",
        "MATIO_LIB_DIR",
        "MATIO_LIB_DIRS",
        "MATIO_LIBS",
        "MATIO_LINK_ARGS",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    build_glaze_helper();
    build_matio_oracle_helper();
}

fn build_glaze_helper() {
    let cpp_src = PathBuf::from("tests/cpp/glaze_interop.cpp");
    if !cpp_src.exists() {
        return;
    }

    match compile_cpp_helper(
        "glaze_interop",
        &cpp_src,
        &[PathBuf::from("reference/glaze/include")],
        None,
    ) {
        Ok(bin_path) => {
            println!("cargo:rustc-env=GLAZE_INTEROP_BIN={}", bin_path.display());
        }
        Err(_) => {
            println!(
                "cargo:warning=Failed to build Glaze interop helper; set CXX or install a C++23 compiler to enable interop tests."
            );
        }
    }
}

fn build_matio_oracle_helper() {
    let cpp_src = PathBuf::from("tests/cpp/matio_oracle.cpp");
    if !cpp_src.exists() {
        return;
    }

    let link_config = match matio_link_config() {
        Ok(config) => config,
        Err(err) => {
            println!(
                "cargo:warning=Skipping MATIO oracle helper; {}. Install matio with pkg-config metadata or set MATIO_INCLUDE_DIRS/MATIO_LIB_DIRS/MATIO_LIBS.",
                sanitize_warning(&err)
            );
            return;
        }
    };

    match compile_cpp_helper("matio_oracle", &cpp_src, &[], Some(&link_config)) {
        Ok(bin_path) => {
            println!("cargo:rustc-env=MATIO_ORACLE_BIN={}", bin_path.display());
        }
        Err(err) => {
            println!(
                "cargo:warning=Failed to build MATIO oracle helper: {}",
                sanitize_warning(&err)
            );
        }
    }
}

fn compile_cpp_helper(
    output_name: &str,
    source: &Path,
    include_paths: &[PathBuf],
    link_config: Option<&NativeLinkConfig>,
) -> Result<PathBuf, String> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|e| e.to_string())?);
    let bin_path = out_dir.join(output_name);
    let compiler = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let mut cmd = Command::new(&compiler);
    cmd.arg("-std=c++23").arg("-O2");

    for include in include_paths {
        cmd.arg("-I").arg(include);
    }

    if let Some(link_config) = link_config {
        for include in &link_config.include_paths {
            cmd.arg("-I").arg(include);
        }
    }

    cmd.arg(source).arg("-o").arg(&bin_path);

    if let Some(link_config) = link_config {
        for link_path in &link_config.link_paths {
            cmd.arg("-L").arg(link_path);
        }
        for framework_path in &link_config.framework_paths {
            cmd.arg("-F").arg(framework_path);
        }
        for lib in &link_config.libs {
            cmd.arg(format!("-l{lib}"));
        }
        for framework in &link_config.frameworks {
            cmd.arg("-framework").arg(framework);
        }
        for arg in &link_config.raw_link_args {
            cmd.arg(arg);
        }
    }

    let output = cmd.output().map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(bin_path)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut message = format!("compiler `{compiler}` exited with {}", output.status);
        if !stderr.trim().is_empty() {
            message.push_str(": ");
            message.push_str(stderr.trim());
        } else if !stdout.trim().is_empty() {
            message.push_str(": ");
            message.push_str(stdout.trim());
        }
        Err(message)
    }
}

fn matio_link_config() -> Result<NativeLinkConfig, String> {
    if let Some(config) = matio_link_config_from_env() {
        return Ok(config);
    }

    let library = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("matio")
        .map_err(|err| err.to_string())?;

    Ok(NativeLinkConfig {
        include_paths: library.include_paths,
        link_paths: library.link_paths,
        libs: library.libs,
        frameworks: library.frameworks,
        framework_paths: library.framework_paths,
        raw_link_args: library.ld_args.into_iter().flatten().collect(),
    })
}

fn matio_link_config_from_env() -> Option<NativeLinkConfig> {
    let mut config = NativeLinkConfig {
        include_paths: env_path_list("MATIO_INCLUDE_DIRS", "MATIO_INCLUDE_DIR"),
        link_paths: env_path_list("MATIO_LIB_DIRS", "MATIO_LIB_DIR"),
        libs: env_string_list("MATIO_LIBS"),
        raw_link_args: env_string_list("MATIO_LINK_ARGS"),
        ..NativeLinkConfig::default()
    };

    if config.include_paths.is_empty()
        && config.link_paths.is_empty()
        && config.libs.is_empty()
        && config.raw_link_args.is_empty()
    {
        return None;
    }

    if config.libs.is_empty() {
        config.libs.push("matio".to_string());
    }
    Some(config)
}

fn env_path_list(list_var: &str, single_var: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = env_string_list(list_var)
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if let Ok(single) = env::var(single_var)
        && !single.trim().is_empty()
    {
        out.push(PathBuf::from(single));
    }
    out
}

fn env_string_list(var: &str) -> Vec<String> {
    env::var(var)
        .ok()
        .map(|value| split_list(&value))
        .unwrap_or_default()
}

fn split_list(raw: &str) -> Vec<String> {
    raw.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn sanitize_warning(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}
