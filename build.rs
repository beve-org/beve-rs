use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Try to build the C++ interop helper with a C++23 compiler (required by Glaze).
    // This is best-effort; tests will skip if not present.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cpp_src = PathBuf::from("tests/cpp/glaze_interop.cpp");
    if !cpp_src.exists() {
        return;
    }

    let mut bin_path = out_dir.clone();
    bin_path.push("glaze_interop");

    // Choose compiler
    let compiler = env::var("CXX").unwrap_or_else(|_| "clang++".to_string());
    let mut cmd = Command::new(compiler);
    cmd.arg("-std=c++23")
        .arg("-O2")
        .arg("-I")
        .arg("reference/glaze/include")
        .arg(&cpp_src)
        .arg("-o")
        .arg(&bin_path);

    match cmd.status() {
        Ok(status) if status.success() => {
            println!("cargo:rustc-env=GLAZE_INTEROP_BIN={}", bin_path.display());
        }
        _ => {
            println!("cargo:warning=Failed to build Glaze interop helper; set CXX or install a C++20 compiler to enable interop tests.");
        }
    }
}
