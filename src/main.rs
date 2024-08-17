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
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        query: "".to_string(),
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
            flags.query = arg.clone();
        }
    }

    flags
}

// create a new file in ~/.local/dewey/queries
// named with the current microsecond timestamp
// containing the user query
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

    if flags.sync {
        no_flags = false;
        ledger::sync_ledger_config()?;
    }

    if flags.embed || flags.full_embed {
        no_flags = false;
        dbio::sync_index(flags.full_embed)?;
    }

    if no_flags {
        if flags.query.is_empty() {
            println!("No flags or query provided, nothing to do");
            info!("No flags or query provided, nothing to do");
            return Ok(());
        }

        let query = flags.query;

        let index = hnsw::HNSW::new()?;
        user_query(&index, query)?;
    }

    Ok(())
}
