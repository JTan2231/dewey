use std::io::{Read, Write};

use dewey_lib::logger::Logger;
use dewey_lib::message::Message;
use dewey_lib::serialization::Serialize;
use dewey_lib::{config, dbio, hnsw, info, ledger};

struct Flags {
    query: String,
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

    for arg in args.iter().skip(1) {
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
                    _ => panic!("Unknown flag: {}", c),
                }
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
    println!("\t-s: Sync kedger with config");
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

        let mut stream = std::net::TcpStream::connect("127.0.0.1:5050")?;
        let query = flags.query;
        let message = Message {
            message_type: "query".to_string(),
            body: query,
        };

        let message_bytes = message.to_bytes();
        stream.write(&message_bytes)?;
        stream.flush()?;

        let mut buffer = [0; 8192];
        stream.read(&mut buffer)?;

        let (response, _) = Message::from_bytes(&buffer, 0)?;
        info!("Received response: {}", response.body);
        println!("\n{}\n", response.body);
    }

    Ok(())
}
