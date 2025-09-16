fn main() {
    let v = vec![true, false, true, true, false];
    let bytes = beve::to_vec(&v).unwrap();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 { print!(" "); }
        print!("{:02x}", b);
    }
    println!();
}

