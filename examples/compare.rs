#![deny(warnings)]

use beve::{Complex, Matrix, MatrixLayout};
use serde::{Deserialize, Serialize};
use std::{
    env,
    error::Error,
    fs::File,
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const WORDS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "eta", "theta"];
const DEFAULT_ITERATIONS: usize = 50;

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    id: u64,
    flag: bool,
    intensity: f64,
    readings: [f32; 4],
    label: String,
    tags: Vec<String>,
}

fn sample_records(count: usize) -> Vec<Record> {
    (0..count)
        .map(|i| Record {
            id: i as u64,
            flag: i % 3 == 0,
            intensity: (i as f64 * 0.125).sin() * (1.0 + (i % 7) as f64),
            readings: [
                i as f32,
                (i as f32 * 0.25).sin(),
                (i as f32 * 0.5).cos(),
                ((i * 31) % 13) as f32,
            ],
            label: format!("sensor-{i:04}"),
            tags: vec![
                format!("batch-{}", i / 32),
                if i % 2 == 0 {
                    "even".into()
                } else {
                    "odd".into()
                },
            ],
        })
        .collect()
}

#[derive(Deserialize)]
struct OwnedMatrix {
    layout: MatrixLayout,
    extents: Vec<usize>,
    value: Vec<f64>,
}

fn sample_matrix(rows: usize, cols: usize) -> (Vec<usize>, Vec<f64>) {
    let extents = vec![rows, cols];
    let data = (0..rows * cols)
        .map(|i| ((i as f64 * 0.03125).sin() + (i as f64 * 0.015625).cos()) * 0.5)
        .collect();
    (extents, data)
}

#[derive(Serialize, Deserialize, Clone)]
struct MixedFrame {
    id: u64,
    scalars: Vec<f64>,
    counts: Vec<u32>,
    complex: Vec<Complex<f64>>,
    notes: String,
    status: Status,
}

#[derive(Serialize, Deserialize, Clone)]
enum Status {
    Idle,
    Running {
        stage: u16,
        gains: Vec<f32>,
    },
    Error {
        code: i32,
        tags: Vec<String>,
        retry: bool,
    },
}

fn mixed_frames(count: usize) -> Vec<MixedFrame> {
    (0..count)
        .map(|i| {
            let scalars = (0..32)
                .map(|j| ((i as f64 * 0.5 + j as f64).sin() * 0.25) + j as f64)
                .collect();
            let counts = (0..32).map(|j| (j as u32) ^ (i as u32)).collect();
            let complex = (0..16)
                .map(|j| Complex {
                    re: (i as f64 * 0.25 + j as f64).cos(),
                    im: (i as f64 * 0.125 + j as f64).sin(),
                })
                .collect();
            let notes = format!("frame-{i:05}");
            let status = match i % 3 {
                0 => Status::Idle,
                1 => Status::Running {
                    stage: (i % 8) as u16,
                    gains: (0..12)
                        .map(|j| (j as f32 * 0.03125).sin() + (i as f32 * 0.0625))
                        .collect(),
                },
                _ => Status::Error {
                    code: -((i % 11) as i32),
                    tags: vec![
                        format!("sensor-{}", WORDS[i % WORDS.len()]),
                        format!("batch-{}", i / 64),
                    ],
                    retry: i % 2 == 0,
                },
            };
            MixedFrame {
                id: i as u64,
                scalars,
                counts,
                complex,
                notes,
                status,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Timing {
    label: &'static str,
    beve: f64,
    serde: f64,
}

struct ScenarioResult {
    scenario: String,
    title: String,
    timings: Vec<Timing>,
}

fn time_many<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed()
}

fn per_iter(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1e6 / iterations as f64
}

fn measure_struct(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let records = sample_records(256);
    let debug = env::var_os("COMPARE_DEBUG").is_some();

    if debug {
        eprintln!("struct: encoding fixture with beve");
    }
    let beve_encoded = beve::to_vec(&records)?;
    if debug {
        eprintln!("struct: encoding fixture with serde-beve");
    }
    let serde_encoded = serde_beve::to_bytes(&records)?;
    if debug {
        eprintln!("struct: fixture encoding done");
    }

    let beve_ser = time_many(iterations, || {
        let bytes = beve::to_vec(black_box(&records)).unwrap();
        black_box(bytes);
    });
    if debug {
        eprintln!("struct: finished beve::to_vec iterations");
    }
    let serde_ser = time_many(iterations, || {
        let bytes = serde_beve::to_bytes(black_box(&records)).unwrap();
        black_box(bytes);
    });
    if debug {
        eprintln!("struct: finished serde_beve::to_bytes iterations");
    }

    let beve_de = time_many(iterations, || {
        let decoded: Vec<Record> = beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    if debug {
        eprintln!("struct: finished beve::from_slice iterations");
    }
    let serde_de = time_many(iterations, || {
        let decoded: Vec<Record> =
            serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    if debug {
        eprintln!("struct: finished serde_beve::from_bytes iterations");
    }

    Ok(vec![
        Timing {
            label: "struct encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "struct decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn measure_numeric(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let values: Vec<f64> = (0..2048)
        .map(|i| ((i as f64 * 0.25).sin() + 1.0) * (i % 1024) as f64)
        .collect();
    let beve_encoded = beve::to_vec(&values)?;
    let serde_encoded = serde_beve::to_bytes(&values)?;

    let beve_ser = time_many(iterations, || {
        let bytes = beve::to_vec(black_box(&values)).unwrap();
        black_box(bytes);
    });
    let serde_ser = time_many(iterations, || {
        let bytes = serde_beve::to_bytes(black_box(&values)).unwrap();
        black_box(bytes);
    });
    let beve_de = time_many(iterations, || {
        let decoded: Vec<f64> = beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    let serde_de = time_many(iterations, || {
        let decoded: Vec<f64> =
            serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(decoded);
    });

    Ok(vec![
        Timing {
            label: "numeric encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "numeric decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn measure_bool(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let flags: Vec<bool> = (0..4096).map(|i| (i % 3 == 0) ^ (i % 5 == 0)).collect();
    let beve_encoded = beve::to_vec(&flags)?;
    let serde_encoded = serde_beve::to_bytes(&flags)?;

    let beve_ser = time_many(iterations, || {
        let bytes = beve::to_vec(black_box(&flags)).unwrap();
        black_box(bytes);
    });
    let serde_ser = time_many(iterations, || {
        let bytes = serde_beve::to_bytes(black_box(&flags)).unwrap();
        black_box(bytes);
    });
    let beve_de = time_many(iterations, || {
        let decoded: Vec<bool> = beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    let serde_de = time_many(iterations, || {
        let decoded: Vec<bool> =
            serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(decoded);
    });

    Ok(vec![
        Timing {
            label: "bool encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "bool decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn measure_strings(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let owned: Vec<String> = (0..1024)
        .map(|i| format!("sensor-{i:04}-{}", WORDS[i % WORDS.len()]))
        .collect();
    let beve_encoded = beve::to_vec(&owned)?;
    let serde_encoded = serde_beve::to_bytes(&owned)?;

    let beve_ser = time_many(iterations, || {
        let bytes = beve::to_vec(black_box(&owned)).unwrap();
        black_box(bytes);
    });
    let serde_ser = time_many(iterations, || {
        let bytes = serde_beve::to_bytes(black_box(&owned)).unwrap();
        black_box(bytes);
    });
    let beve_de = time_many(iterations, || {
        let decoded: Vec<String> = beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    let serde_de = time_many(iterations, || {
        let decoded: Vec<String> =
            serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(decoded);
    });

    Ok(vec![
        Timing {
            label: "string encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "string decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn measure_matrix(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let (extents, data) = sample_matrix(48, 48);
    let matrix = Matrix {
        layout: MatrixLayout::Right,
        extents: &extents,
        data: &data,
    };
    let beve_encoded = beve::to_vec(&matrix)?;
    let serde_encoded = serde_beve::to_bytes(&matrix)?;

    let beve_ser = time_many(iterations, || {
        let matrix = Matrix {
            layout: MatrixLayout::Right,
            extents: &extents,
            data: &data,
        };
        let bytes = beve::to_vec(black_box(&matrix)).unwrap();
        black_box(bytes);
    });
    let serde_ser = time_many(iterations, || {
        let matrix = Matrix {
            layout: MatrixLayout::Right,
            extents: &extents,
            data: &data,
        };
        let bytes = serde_beve::to_bytes(black_box(&matrix)).unwrap();
        black_box(bytes);
    });
    let beve_de = time_many(iterations, || {
        let OwnedMatrix {
            layout,
            extents,
            value,
        } = beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(layout);
        black_box(extents);
        black_box(value);
    });
    let serde_de = time_many(iterations, || {
        let OwnedMatrix {
            layout,
            extents,
            value,
        } = serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(layout);
        black_box(extents);
        black_box(value);
    });

    Ok(vec![
        Timing {
            label: "matrix encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "matrix decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn measure_mixed(iterations: usize) -> Result<Vec<Timing>, Box<dyn Error>> {
    let frames = mixed_frames(128);
    let beve_encoded = beve::to_vec(&frames)?;
    let serde_encoded = serde_beve::to_bytes(&frames)?;

    let beve_ser = time_many(iterations, || {
        let bytes = beve::to_vec(black_box(&frames)).unwrap();
        black_box(bytes);
    });
    let serde_ser = time_many(iterations, || {
        let bytes = serde_beve::to_bytes(black_box(&frames)).unwrap();
        black_box(bytes);
    });
    let beve_de = time_many(iterations, || {
        let decoded: Vec<MixedFrame> =
            beve::from_slice(black_box(beve_encoded.as_slice())).unwrap();
        black_box(decoded);
    });
    let serde_de = time_many(iterations, || {
        let decoded: Vec<MixedFrame> =
            serde_beve::from_bytes(black_box(serde_encoded.as_slice())).unwrap();
        black_box(decoded);
    });

    Ok(vec![
        Timing {
            label: "mixed encode",
            beve: per_iter(beve_ser, iterations),
            serde: per_iter(serde_ser, iterations),
        },
        Timing {
            label: "mixed decode",
            beve: per_iter(beve_de, iterations),
            serde: per_iter(serde_de, iterations),
        },
    ])
}

fn print_timings(title: &str, timings: &[Timing]) {
    use std::fmt::Write as _;

    println!("\n== {title} ==");
    for timing in timings {
        let mut line = String::new();
        let _ = write!(
            &mut line,
            "{:<16} beve {:>8.2} µs | serde-beve {:>8.2} µs",
            timing.label, timing.beve, timing.serde
        );
        if timing.serde > 0.0 {
            let ratio = timing.serde / timing.beve;
            let _ = write!(&mut line, "  (serde / beve = {:>5.2}x)", ratio);
        }
        println!("{line}");
    }
}

fn write_csv(path: &Path, results: &[ScenarioResult]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "scenario,title,label,beve_us,serde_beve_us,serde_over_beve"
    )?;

    for result in results {
        for timing in &result.timings {
            let ratio = if timing.beve > 0.0 {
                timing.serde / timing.beve
            } else {
                f64::NAN
            };
            writeln!(
                file,
                "{},{},{},{:.6},{:.6},{:.6}",
                result.scenario, result.title, timing.label, timing.beve, timing.serde, ratio
            )?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut iterations = DEFAULT_ITERATIONS;
    let mut requested = Vec::new();
    let mut csv_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => {
                let value = args.next().ok_or_else(|| "expected value after --iters")?;
                iterations = value.parse()?;
            }
            "--csv" => {
                let value = args.next().ok_or_else(|| "expected path after --csv")?;
                csv_path = Some(value.into());
            }
            other => requested.push(other.to_string()),
        }
    }

    if iterations == 0 {
        return Err("iteration count must be > 0".into());
    }

    if requested.is_empty() {
        requested.extend(
            ["struct", "numeric", "bool", "string", "matrix", "mixed"]
                .iter()
                .map(|s| s.to_string()),
        );
    }

    let mut collected = Vec::new();

    for request in &requested {
        let (title, timings) = match request.as_str() {
            "struct" => ("Struct Records", measure_struct(iterations)?),
            "numeric" => ("Numeric f64 Arrays", measure_numeric(iterations)?),
            "bool" => ("Bool Arrays", measure_bool(iterations)?),
            "string" => ("String Arrays", measure_strings(iterations)?),
            "matrix" => ("Matrix Payloads", measure_matrix(iterations)?),
            "mixed" => ("Mixed Frames", measure_mixed(iterations)?),
            other => {
                eprintln!("Unknown scenario '{other}', skipping");
                continue;
            }
        };

        print_timings(title, &timings);
        collected.push(ScenarioResult {
            scenario: request.clone(),
            title: title.to_string(),
            timings,
        });
    }

    if let Some(path) = csv_path {
        write_csv(&path, &collected)?;
        println!("\nWrote CSV summary to {}", path.display());
    }

    Ok(())
}
