mod config;
mod dbio;
mod hnsw;
mod ledger;
mod logger;
mod openai;

use crate::logger::Logger;

struct Flags {
    query: String,
    sync: bool,
    embed: bool,
    full_embed: bool,
    reindex: bool,
    help: bool,
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
                    'r' => flags.reindex = true,
                    'h' => flags.help = true,
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
    println!("Usage: dewey [-sef] [query]");
    println!("\t-s: sync ledger with config");
    println!("\t-e: embed stale files");
    println!("\t-f: force re-embed all files");
    println!("\t-r: reindex the database");
}

fn user_query(index: &hnsw::HNSW, query: String) -> Result<(), std::io::Error> {
    let timestamp = chrono::Utc::now().timestamp_micros();
    let path = config::get_local_dir()
        .join("queries")
        .join(timestamp.to_string());
    std::fs::write(path.clone(), query)?;
    info!("Wrote query to {}", path.to_string_lossy());

    let embedding = openai::embed(&vec![openai::EmbeddingSource {
        filepath: path.to_string_lossy().to_string(),
        subset: None,
    }])?;

    let result = index.query(&embedding[0], 10, 30);

    for (e, d) in result {
        println!("{}: {}", e.source_file.filepath, d);
        info!("{}: {}", e.source_file.filepath, d);
    }

    Ok(())
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

    if no_flags {
        if flags.query.is_empty() {
            println!("No flags or query provided, nothing to do");
            info!("No flags or query provided, nothing to do");
            return Ok(());
        }

        let query = flags.query;

        let index = hnsw::HNSW::new(false)?;
        user_query(&index, query)?;
    }

    Ok(())
}
