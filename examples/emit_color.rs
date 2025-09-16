use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Color { Red, Green, Blue }

fn main() {
    let bytes = beve::to_vec(&Color::Green).unwrap();
    use std::io::Write;
    let mut stdout = std::io::stdout();
    stdout.write_all(&bytes).unwrap();
}

