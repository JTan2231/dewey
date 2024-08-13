use sha2::digest::Update;
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};

use crate::logger::Logger;
use crate::{error, info};

const WHITELIST: &[&str] = &[
    "c", "cpp", "cxx", "h", "hpp", "java", "class", "py", "pyw", "pyx", "js", "jsx", "ts", "tsx",
    "rb", "erb", "php", "phtml", "cs", "go", "rs", "swift", "kt", "kts", "scala", "sc", "html",
    "htm", "css", "sass", "scss", "sh", "bash", "zsh", "pl", "pm", "lua", "hs", "lhs", "lisp",
    "cl", "el", "dart", "r", "R", "jl", "groovy", "m", "mm", "asm", "s", "f", "for", "f90", "f95",
    "vb", "bas", "erl", "hrl", "ada", "adb", "ads", "clj", "cljs", "cljc", "fs", "fsx", "cob",
    "cbl", "pas", "pp", "pro", "scm", "ss", "tcl", "v", "vh", "vhd", "vhdl", "xml", "md",
    "markdown", "ipynb", "ps1", "psm1", "psd1", "bat", "cmd", "elm", "ex", "exs", "ml", "mli",
    "mat", "sql",
];

#[derive(Debug)]
struct LedgerEntry {
    filepath: String,
    hash: String,
}

fn read_ledger() -> Vec<LedgerEntry> {
    let ledger_path = crate::config::get_local_dir().join("ledger");
    let ledger_file = std::fs::File::open(&ledger_path).expect("Failed to open ledger file");

    let mut reader = std::io::BufReader::new(ledger_file);
    let mut entries = Vec::new();
    let mut line = String::new();
    while reader.read_line(&mut line).is_ok() {
        if line.is_empty() {
            break;
        }

        let parts: Vec<&str> = line.split_whitespace().filter(|s| !s.is_empty()).collect();
        if parts.len() == 2 {
            entries.push(LedgerEntry {
                filepath: parts[0].to_string(),
                hash: parts[1].to_string(),
            });
        } else {
            panic!("Malformed ledger entry: {:?}", parts);
        }

        line.clear();
    }

    entries
}

// returns a list of files whose hashes are out of date with file contents
pub fn get_stale_files() -> Vec<String> {
    let ledger = read_ledger();
    let mut stale_files = Vec::new();
    for entry in ledger.iter() {
        let hash = get_hash(&entry.filepath);
        if true
        /*hash != entry.hash*/
        {
            stale_files.push(entry.filepath.clone());
        }
    }

    stale_files
}

fn is_whitelisted(path: &str) -> bool {
    for ext in WHITELIST {
        if path.ends_with(format!(".{}", ext).as_str()) {
            return true;
        }
    }

    false
}

fn get_hash(filepath: &String) -> String {
    let content = std::fs::read(filepath).expect("Failed to read file");
    let mut hasher = Sha256::new();
    Update::update(&mut hasher, &content);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

// current functionality is that it uses .gitignore files
// to blacklist files that are under the directories
// in the ledger config
//
// this should probably be expanded in the future
//
// this could also probably be optimized
pub fn sync_ledger_config() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = crate::config::get_config_dir();
    let config_ledger_path = config_path.join("ledger");

    let config_ledger = std::fs::read_to_string(&config_ledger_path)?;
    let config_ledger = config_ledger
        .lines()
        .filter(|line| {
            let parts: Vec<&str> = line.split_whitespace().filter(|s| !s.is_empty()).collect();
            let cond = parts.len() == 1;
            if !cond {
                error!("Ignoring malformed ledger entry: {}", line);
            }

            cond
        })
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let mut gitignores = Vec::new();
    let mut config_entries = Vec::new();
    for mut entry in config_ledger {
        let path = std::path::Path::new(&entry);
        if path.is_dir() && (!entry.ends_with("*") || !entry.ends_with("**")) {
            entry.push_str("/**/*");
        }

        for file in glob::glob(&entry).expect("Failed to read glob pattern") {
            let file = file?;
            if is_whitelisted(file.to_str().unwrap()) {
                config_entries.push(file.to_string_lossy().to_string());
            }

            if file.ends_with(".gitignore") {
                gitignores.push(file.to_string_lossy().to_string());
            }
        }
    }

    let mut gitignore_globs = Vec::new();
    for gitignore in gitignores {
        let file = std::fs::File::open(&gitignore)?;
        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("#") || line.is_empty() {
                continue;
            }

            if line.starts_with("!") {
                continue;
            }

            let line = match line.strip_prefix("/") {
                Some(line) => line,
                None => line.as_str(),
            };

            let full_path = std::path::Path::new(&gitignore)
                .parent()
                .unwrap()
                .join(line);

            let full_path = match full_path.is_dir() {
                true => format!("{}/**/*", full_path.to_string_lossy()),
                false => full_path.to_string_lossy().to_string(),
            };
            gitignore_globs.push(full_path);
        }
    }

    match std::env::var("GOPATH") {
        Ok(_) => {
            gitignore_globs.push("pkg/**/*".to_string());
        }
        Err(_) => {
            let home = crate::config::get_home_dir();
            gitignore_globs.push(format!("{}/go/pkg/**/*", home.to_string_lossy()));
        }
    }

    let mut ignored_count = 0;
    config_entries = config_entries
        .into_iter()
        .filter(|entry| {
            for glob in gitignore_globs.iter() {
                if glob::Pattern::new(glob).unwrap().matches(entry) {
                    ignored_count += 1;
                    return false;
                }
            }
            true
        })
        .collect();

    let mut local_ledger = read_ledger();

    // n^2 lol
    let mut new_entries = Vec::new();
    for entry in config_entries.iter() {
        if !local_ledger.iter().any(|e| e.filepath == *entry) {
            new_entries.push(entry);
        }
    }

    info!("Adding {} new entries to ledger", new_entries.len());
    info!("Ignoring {} entries", ignored_count);
    for new_entry in new_entries {
        info!("Adding new entry to ledger: {}", new_entry);
        let entry = LedgerEntry {
            filepath: new_entry.to_string(),
            hash: "0".to_string(),
        };

        local_ledger.push(entry);
    }

    for entry in local_ledger.iter_mut() {
        entry.hash = get_hash(&entry.filepath);
    }

    info!("Final ledger size: {}", local_ledger.len());

    match std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(crate::config::get_local_dir().join("ledger"))
    {
        Ok(mut file) => {
            for entry in local_ledger {
                writeln!(file, "{} {}", entry.filepath, entry.hash)?;
            }
        }
        Err(e) => {
            error!("Failed to write ledger file: {}", e);
        }
    }

    Ok(())
}
