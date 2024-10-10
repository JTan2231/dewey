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
    println!("Usage: dewey [-sefrhb] [query]");
    println!("\nFlags:");
    println!("\t-s: Sync ledger with config");
    println!("\t-e: Embed missing items in ledger");
    println!("\t-f: Embed all items in ledger");
    println!("\t-r: Reindex embeddings");
    println!("\t-h: Print this help message");
    println!("\t-b: Reblock embeddings");
    println!("\nQuery:");
    println!("\tQuery to send to the server");
    println!("\nExamples:");
    println!("\tdewey -ser");
    println!("\tdewey -serb");
    println!("\tdewey -sfrb \"query\"");
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
        if flags.query.is_empty() {
            println!("No flags or query provided, nothing to do");
            info!("No flags or query provided, nothing to do");
            return Ok(());
        }

        let mut stream = std::net::TcpStream::connect("127.0.0.1:5051")?;

        let message = DeweyRequest {
            query: flags.query,
            filters: flags.query_filters,
        };

        let message_bytes = serde_json::to_string(&message)?.into_bytes();
        stream.write(&message_bytes)?;
        stream.flush()?;

        let mut length_bytes = [0u8; 4];
        stream.read_exact(&mut length_bytes)?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        let mut buffer = vec![0u8; length];
        stream.read_exact(&mut buffer)?;
        let buffer = String::from_utf8_lossy(&buffer);

        let response: DeweyResponse = match serde_json::from_str(&buffer) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Failed to parse response: {}", e);
                error!("buffer: {:?}", buffer);
                return Err(e.into());
            }
        };

        info!("Received response: {}", response.body);
        println!("\n{}\n", response.body);
    }

    Ok(())
}
