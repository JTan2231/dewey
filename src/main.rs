mod btree;
mod config;
mod ledger;
mod logger;
mod openai;

struct Flags {
    sync: bool,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags { sync: false };

    if args.len() < 2 {
        panic!("Usage: {} [-S]", args[0]);
    }

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-S" => flags.sync = true,
            _ => panic!("Unknown flag: {}", arg),
        }
    }

    flags
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    config::setup();
    let flags = parse_flags();

    if flags.sync {
        ledger::sync_ledger_config()?;
    }

    Ok(())
}
