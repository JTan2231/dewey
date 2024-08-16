mod config;
mod dbio;
mod hnsw;
mod ledger;
mod logger;
mod openai;

use crate::logger::Logger;
use crate::openai::Embedding;
use rand::Rng;

struct Flags {
    sync: bool,
    embed: bool,
    full_embed: bool,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        sync: false,
        embed: false,
        full_embed: false,
    };

    if args.len() < 1 {
        panic!("Usage: {} [-sef]", args[0]);
    }

    for arg in args.iter().skip(1) {
        if arg.starts_with("-") && !arg.starts_with("--") {
            for c in arg.chars().skip(1) {
                match c {
                    's' => flags.sync = true,
                    'e' => flags.embed = true,
                    'f' => flags.full_embed = true,
                    _ => panic!("Unknown flag: {}", c),
                }
            }
        } else {
            panic!("Unknown argument: {}", arg);
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

    if flags.embed || flags.full_embed {
        dbio::sync_index(flags.full_embed)?;
    }

    if !flags.sync && !flags.embed {
        let hnsw = hnsw::HNSW::new(0);
        hnsw.print_graph();

        let query = openai::embed(&vec![openai::EmbeddingSource {
            filepath: "src/testing".to_string(),
            subset: None,
        }])?;

        let query = query[0].clone();
        let result = hnsw.query(&query, 10, 30);

        info!(
            "{:?}",
            result
                .clone()
                .into_iter()
                .map(|(e, d)| (e.source_file.filepath, d))
                .collect::<Vec<_>>()
        );
    }

    Ok(())
}
