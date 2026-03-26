use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=tests/cpp/glaze_interop.cpp");
    println!("cargo:rerun-if-changed=tests/cpp/glaze_bench.cpp");
    println!("cargo:rerun-if-changed=reference/glaze/include");

    for var in [
        "CXX",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    build_glaze_helper();
    build_glaze_bench();
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

fn build_glaze_bench() {
    let cpp_src = PathBuf::from("tests/cpp/glaze_bench.cpp");
    if !cpp_src.exists() {
        return;
    }

    match compile_cpp_helper(
        "glaze_bench",
        &cpp_src,
        &[PathBuf::from("reference/glaze/include")],
    ) {
        Ok(bin_path) => {
            println!("cargo:rustc-env=GLAZE_BENCH_BIN={}", bin_path.display());
        }
        Err(_) => {
            // Silently skip — the interop warning is sufficient
        }
    }
}

fn compile_cpp_helper(
    output_name: &str,
    source: &Path,
    include_paths: &[PathBuf],
) -> Result<PathBuf, String> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|e| e.to_string())?);
    let bin_path = out_dir.join(output_name);
    let compiler = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let mut cmd = Command::new(&compiler);
    cmd.arg("-std=c++23").arg("-O2");

    for include in include_paths {
        cmd.arg("-I").arg(include);
    }

    cmd.arg(source).arg("-o").arg(&bin_path);

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
