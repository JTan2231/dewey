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
    "mat", "sql", "txt",
];

#[derive(Debug)]
pub struct LedgerEntry {
    pub filepath: String,
    pub hash: String,
}

pub fn read_ledger() -> Vec<LedgerEntry> {
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
pub fn get_stale_files() -> Result<Vec<String>, std::io::Error> {
    let ledger = read_ledger();
    let mut stale_files = Vec::new();
    for entry in ledger.iter() {
        let hash = get_hash(&entry.filepath)?;
        if hash != entry.hash {
            stale_files.push(entry.filepath.clone());
        }
    }

    Ok(stale_files)
}

fn is_whitelisted(path: &str) -> bool {
    for ext in WHITELIST {
        if path.ends_with(format!(".{}", ext).as_str()) {
            return true;
        }
    }

    false
}

fn get_hash(filepath: &String) -> Result<String, std::io::Error> {
    let content = std::fs::read(filepath)?;
    let mut hasher = Sha256::new();
    Update::update(&mut hasher, &content);
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>())
}

// current functionality is that it uses .gitignore files
// to blacklist files that are under the directories
// in the ledger config
//
// this should probably be expanded in the future
//
// this could also probably be optimized
//
// this function rebuilds the `~/.local/dewey/ledger` file
// according to what's in `~/.config/dewey/ledger`
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

    let mut config_entries = Vec::new();
    for mut entry in config_ledger {
        let path = std::path::Path::new(&entry);
        if path.is_dir() && (!entry.ends_with("*") || !entry.ends_with("**")) {
            entry.push_str("/**/*");
        }

        info!("searching for files in {}", entry);

        let directory = glob::glob(&entry)
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        // there has to be a better way of dealing with go pkg directories than this
        let mut gitignore_globs = vec!["pkg/mod/**/*".to_string()];
        for file in directory.iter() {
            if file.ends_with(".gitignore") {
                let gitignore = file.clone();
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
        }

        let mut kept = 0;
        config_entries.extend(
            directory
                .iter()
                .filter(|f| {
                    for glob in gitignore_globs.iter() {
                        if glob::Pattern::new(glob)
                            .unwrap()
                            .matches(f.to_str().unwrap())
                            || f.to_str().unwrap().contains("pkg/mod")
                        {
                            return false;
                        }
                    }

                    if is_whitelisted(f.to_str().unwrap()) {
                        kept += 1;
                        return true;
                    } else {
                        return false;
                    }
                })
                .map(|f| f.to_string_lossy().to_string()),
        );

        info!("Kept {} files from {}", kept, entry);
        println!("Kept {} files from {}", kept, entry);
    }

    info!("{} config entries", config_entries.len());

    let new_ledger = config_entries
        .into_iter()
        .map(|s| LedgerEntry {
            filepath: s.clone(),
            hash: get_hash(&s).unwrap(),
        })
        .collect::<Vec<_>>();

    info!("New ledger size: {}", new_ledger.len());
    println!("New ledger size: {}", new_ledger.len());

    match std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(crate::config::get_local_dir().join("ledger"))
    {
        Ok(mut file) => {
            for entry in new_ledger {
                writeln!(file, "{} {}", entry.filepath, entry.hash)?;
            }
        }
        Err(e) => {
            error!("Failed to write ledger file: {}", e);
        }
    }

    Ok(())
}
