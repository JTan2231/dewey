use std::io::{Read, Write};

use dewey_lib::logger::Logger;
use dewey_lib::message::{DeweyRequest, DeweyResponse};
use dewey_lib::{config, dbio, error, hnsw, info, ledger};

struct Flags {
    query: String,
    query_filters: Vec<String>,
    sync: bool,
    embed: bool,
    full_embed: bool,
    reindex: bool,
    help: bool,
    test: bool,
    reblock: bool,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        query: "".to_string(),
        query_filters: Vec::new(),
        sync: false,
        embed: false,
        full_embed: false,
        reindex: false,
        help: false,
        test: false,
        reblock: false,
    };

    if args.len() < 1 {
        man();
        std::process::exit(1);
    }

    for (i, arg) in args.iter().skip(1).enumerate() {
        if arg.starts_with("-") && !arg.starts_with("--") {
            for c in arg.chars().skip(1) {
                match c {
                    's' => flags.sync = true,
                    'e' => flags.embed = true,
                    'f' => flags.full_embed = true,
                    'r' => flags.reindex = true,
                    'h' => flags.help = true,
                    't' => flags.test = true,
                    'b' => flags.reblock = true,
                    _ => panic!("error: unknown flag: {}", c),
                }
            }
        } else if arg.starts_with("--") {
            match arg.as_str() {
                "--filter" => {
                    if let Some(filter_value) = args.get(i + 1) {
                        if filter_value.matches(",").count() != 1 {
                            panic!(
                                "error: malformed filter value, expected format is 'field,value'"
                            );
                        }

                        flags.query_filters.push(filter_value.clone());
                    } else {
                        panic!("error: missing filter value after --filter");
                    }
                }
                _ => panic!("error: unknown flag: {}", arg),
            }
        } else {
            flags.query = arg.clone();
        }
    }

    flags
}

fn man() {
    println!("Dewey - A simple document retrieval system");
    println!("Usage: dewey [OPTIONS] [QUERY]");
    println!("");
    println!("Options:");
    println!("  -s        Synchronize the ledger with the configuration");
    println!("  -e        Embed missing items in the ledger");
    println!("  -f        Embed all items in the ledger (re-embed)");
    println!("  -r        Reindex embeddings");
    println!("  -b        Reblock embeddings");
    println!("  -h        Print this help message");
    println!("");
    println!("Arguments:");
    println!("");
    println!("Examples:");
    println!("  dewey -s                Synchronize the ledger");
    println!("  dewey -e                Embed missing items");
    println!("  dewey -r                Reindex the embeddings");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    config::setup();
    let flags = parse_flags();
    let mut no_flags = true;

    if flags.help {
        man();
        return Ok(());
    }

    if flags.sync {
        no_flags = false;
        ledger::sync_ledger_config()?;
    }

    if flags.embed || flags.full_embed {
        no_flags = false;
        dbio::sync_index(flags.full_embed)?;
    }

    if flags.reindex {
        no_flags = false;
        let index = hnsw::HNSW::new(true)?;

        let data_dir = config::get_data_dir();
        index.serialize(&data_dir.join("index").to_str().unwrap().to_string())?;
    }

    if flags.reblock {
        dbio::reblock()?;
    }

    if no_flags {
        println!("No flags provided, nothing to do");
        info!("No flags provided, nothing to do");
        return Ok(());
    }

    Ok(())
}
