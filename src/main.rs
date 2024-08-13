/*mod btree;
mod dbio;*/
mod config;
mod ledger;
mod logger;
mod openai;

use crate::logger::Logger;

struct Flags {
    sync: bool,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags { sync: false };

    if args.len() < 1 {
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
    } else {
        let stale_files = ledger::get_stale_files();
        let stale_sources = stale_files
            .iter()
            .map(|f| openai::EmbeddingSource {
                filepath: f.clone(),
                subset: None,
            })
            .collect::<Vec<_>>();

        let embeddings = openai::embed(&stale_sources)?;
        info!("Embeddings: {}", embeddings.len());
    }

    Ok(())
}
