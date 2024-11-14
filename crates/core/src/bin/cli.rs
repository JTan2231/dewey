use dewey_lib::logger::Logger;
use dewey_lib::lprint;
use dewey_lib::{config, dbio, hnsw, info, ledger};

struct Flags {
    query: String,
    query_filters: Vec<String>,
    sync: bool,
    embed: bool,
    full_embed: bool,
    reindex: bool,
    help: bool,
    full_help: bool,
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
        full_help: false,
        test: false,
        reblock: false,
    };

    if args.len() < 1 {
        usage();
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
                    'H' => flags.full_help = true,
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
    println!("\x1b[1mDEWEY(1)\x1b[0m                          User Commands                          \x1b[1mDEWEY(1)\x1b[0m\n");
    println!("\x1b[1mNAME\x1b[0m");
    println!("    dewey - An efficient document retrieval and embedding system\n");

    println!("\x1b[1mSYNOPSIS\x1b[0m");
    println!("    \x1b[1mdewey\x1b[0m [\x1b[4mOPTIONS\x1b[0m] [\x1b[4mQUERY\x1b[0m]\n");

    println!("\x1b[1mDESCRIPTION\x1b[0m");
    println!("    Dewey is a document retrieval system that uses embeddings to enable semantic");
    println!("    search across your document collection. It maintains a ledger of documents and");
    println!("    their embeddings for efficient querying.\n");

    println!("\x1b[1mOPTIONS\x1b[0m");
    println!("    \x1b[1m-s\x1b[0m, \x1b[1m--sync\x1b[0m");
    println!("        Synchronize the ledger with the configuration file. This updates the");
    println!("        document collection based on your current configuration.\n");

    println!("    \x1b[1m-e\x1b[0m, \x1b[1m--embed\x1b[0m");
    println!(
        "        Generate embeddings for any documents in the ledger that don't have them yet.\n"
    );

    println!("    \x1b[1m-f\x1b[0m, \x1b[1m--full-embed\x1b[0m");
    println!("        Regenerate embeddings for all documents in the ledger, regardless of");
    println!("        whether they already have embeddings.\n");

    println!("    \x1b[1m-r\x1b[0m, \x1b[1m--reindex\x1b[0m");
    println!("        Rebuild the search index using the current embeddings. This can improve");
    println!("        search performance.\n");

    println!("    \x1b[1m-b\x1b[0m, \x1b[1m--reblock\x1b[0m");
    println!("        Reorganize the embedding blocks for optimal performance.\n");

    println!("    \x1b[1m--filter\x1b[0m \x1b[4mFIELD,VALUE\x1b[0m");
    println!("        Filter search results based on document metadata. Format: field,value\n");

    println!("    \x1b[1m-h\x1b[0m, \x1b[1m--help\x1b[0m");
    println!("        Display this help message and exit.\n");

    println!("\x1b[1mEXAMPLES\x1b[0m");
    println!("    Initialize and prepare the system:");
    println!("        \x1b[1mdewey -s -e\x1b[0m");
    println!("            Sync the ledger and generate missing embeddings\n");

    println!("    Search with filters:");
    println!("        \x1b[1mdewey \"machine learning\" --filter type,research\x1b[0m");
    println!("            Search for \"machine learning\" in research documents\n");

    println!("    Maintenance operations:");
    println!("        \x1b[1mdewey -r -b\x1b[0m");
    println!("            Reindex and reblock for optimal performance\n");
}

fn usage() {
    println!("Usage: dewey [OPTIONS] [QUERY]\n");
    println!("Options:");
    println!("  -s         sync ledger with config");
    println!("  -e         embed missing documents");
    println!("  -f         re-embed all documents");
    println!("  -r         rebuild search index");
    println!("  -b         reblock embeddings");
    println!("  --filter   field,value  filter results");
    println!("  -h         show this message\n");
    println!("  -H         show full help message\n");
    println!("Example: dewey -se \"machine learning\"");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    config::setup();
    let flags = parse_flags();
    let mut no_flags = true;

    lprint!(
        info,
        "Compiled for regression testing: {}",
        cfg!(feature = "regression")
    );

    if flags.help {
        usage();
        return Ok(());
    }

    if flags.full_help {
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
