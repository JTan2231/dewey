use sha2::digest::Update;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, Write};

use crate::logger::Logger;
use crate::{error, info};

// TODO: there needs to be better delineation on the different rule types
//       Currently, MinLength and Alphanumeric act as filters,
//       while the rest act as splitting rules.
//       Filters are applied _only_ after splitting rules.
#[derive(Debug, PartialEq, Clone)]
pub enum IndexRuleType {
    Split,
    Naive,
    Code,
    MinLength,
    MaxLength,
    Alphanumeric,
}

impl IndexRuleType {
    pub fn validate(&self, value: &str) -> bool {
        match self {
            IndexRuleType::MinLength => {
                if value.parse::<usize>().is_err() {
                    error!("Ignoring invalid min length value: {}", value);
                    false
                } else {
                    true
                }
            }
            IndexRuleType::MaxLength => {
                if value.parse::<usize>().is_err() {
                    error!("Ignoring invalid max length value: {}", value);
                    false
                } else {
                    true
                }
            }
            IndexRuleType::Alphanumeric => {
                if value.to_lowercase() != "true" && value.to_lowercase() != "false" {
                    error!("Ignoring invalid alphanumeric value: {}", value);
                    false
                } else {
                    true
                }
            }
            IndexRuleType::Split => {
                if value.len() == 0 {
                    error!("Ignoring invalid empty split value");
                    false
                } else {
                    true
                }
            }

            IndexRuleType::Code => {
                if value.to_lowercase() != "function" {
                    error!("Ignoring invalid code value: {}", value);
                    false
                } else {
                    true
                }
            }
            _ => true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexRule {
    pub rule_type: IndexRuleType,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub filepath: String,
    pub hash: String,
    pub meta: std::collections::HashSet<String>,
}

pub fn read_ledger() -> Result<Vec<LedgerEntry>, std::io::Error> {
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
        if parts
            .get(0)
            .map_or(false, |path| std::path::Path::new(path).exists())
        {
            entries.push(LedgerEntry {
                filepath: parts[0].to_string(),
                hash: parts[1].to_string(),
                meta: parts[2]
                    .split(",")
                    .map(|s| s.to_string())
                    .collect::<std::collections::HashSet<String>>(),
            });
        } else {
            panic!("Malformed ledger entry: {:?}", parts);
        }

        line.clear();
    }

    Ok(entries)
}

// returns a list of files whose hashes are out of date with file contents
pub fn get_stale_files() -> Result<Vec<LedgerEntry>, std::io::Error> {
    let ledger = read_ledger()?;
    let mut stale_files = Vec::new();
    for entry in ledger.iter() {
        let hash = get_hash(&entry.filepath)?;
        if hash != entry.hash {
            stale_files.push(entry.clone());
        }
    }

    Ok(stale_files)
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

// the rules config is housed in ~/.config/dewey/rules
// each rule has its own line and is formatted like so:
//   `extension --rule_type value --rule_type value ...`
// where:
//   - `extension` is the file extension to which the rule applies
//   - `rule_type` is the type of rule to apply
//   - `value` is the value of the rule
pub fn get_indexing_rules() -> Result<HashMap<String, Vec<IndexRule>>, std::io::Error> {
    let config_path = crate::config::get_config_dir();
    let config_index_path = config_path.join("rules");

    let file = std::fs::File::open(&config_index_path)?;
    let reader = std::io::BufReader::new(file);
    let mut rulesets = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            error!("Ignoring malformed index rule: {}", line);
            continue;
        }

        let extension = parts[0].to_string();
        let mut rule = IndexRule {
            rule_type: IndexRuleType::Naive,
            value: "".to_string(),
        };

        let mut rules = Vec::new();
        for part in parts.iter().skip(1) {
            if part.starts_with("--") {
                match part.to_lowercase().as_str() {
                    "--code" => rule.rule_type = IndexRuleType::Code,
                    "--split" => rule.rule_type = IndexRuleType::Split,
                    "--maxlength" => rule.rule_type = IndexRuleType::MaxLength,
                    "--minlength" => rule.rule_type = IndexRuleType::MinLength,
                    "--alphanumeric" => rule.rule_type = IndexRuleType::Alphanumeric,
                    _ => {
                        error!("Ignoring unknown rule type: {}", part);
                    }
                }
            } else {
                rule.value = part.to_string();
                rule.value = rule
                    .value
                    .replace("\"", "")
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace("\\r", "\r");

                if !rule.rule_type.validate(&rule.value) {
                    continue;
                }

                rules.push(rule);

                rule = IndexRule {
                    rule_type: IndexRuleType::Naive,
                    value: "".to_string(),
                };
            }
        }

        rulesets.insert(extension, rules);
    }

    Ok(rulesets)
}

struct ConfigEntry {
    pub filepath: String,
    pub meta: std::collections::HashSet<String>,
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
//
// files in the config ledger can be commented out with `#`
pub fn sync_ledger_config() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = crate::config::get_config_dir();
    let config_ledger_path = config_path.join("ledger");

    let config_ledger = std::fs::read_to_string(&config_ledger_path)?;
    let mut config_ledger = config_ledger
        .lines()
        .filter(|line| {
            let parts: Vec<&str> = line.split_whitespace().filter(|s| !s.is_empty()).collect();
            let cond = parts
                .get(0)
                .map_or(false, |path| std::path::Path::new(path).exists())
                && parts.iter().skip(1).all(|&s| s.starts_with("--"));

            if !cond {
                error!("Ignoring malformed ledger entry: {}", line);
            }

            cond
        })
        .map(|line| {
            let parts = line
                .split_whitespace()
                .filter(|s| !s.is_empty())
                .collect::<Vec<&str>>();
            let filepath = parts[0];

            let mut meta = std::collections::HashSet::new();
            for part in parts.iter().skip(1) {
                meta.insert(part.to_string());
            }

            ConfigEntry {
                filepath: filepath.to_string(),
                meta,
            }
        })
        .collect::<Vec<_>>();

    let mut meta_index = 0;
    let mut config_entries = Vec::new();
    for config_entry in config_ledger.iter_mut() {
        let entry = &mut config_entry.filepath;
        if entry.starts_with("#") {
            continue;
        }

        let path = std::path::Path::new(&entry);
        if path.is_dir() && (!entry.ends_with("*") || !entry.ends_with("**")) {
            if entry.ends_with("/") {
                entry.push_str("**/*");
            } else {
                entry.push_str("/**/*");
            }
        }

        info!("searching for files in {}", entry);

        let directory = glob::glob(&entry)
            .expect("Failed to read glob pattern")
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        // there has to be a better way of dealing with go pkg directories than this
        let mut gitignore_globs = Vec::new();
        for file in directory.iter() {
            if file.ends_with(".gitignore") {
                let gitignore = file.clone();
                let file = std::fs::File::open(&gitignore)?;
                let reader = std::io::BufReader::new(file);

                let root = std::path::Path::new(&gitignore).parent().unwrap();
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

                    let full_path = root.join(line);
                    let is_dir = match glob::glob(&full_path.to_string_lossy().to_string()) {
                        Ok(matches) => matches.peekable().any(|m| m.is_ok() && m.unwrap().is_dir()),
                        Err(_) => false,
                    } || full_path.is_dir();

                    let full_path = full_path.to_string_lossy().to_string();

                    let full_path = match is_dir {
                        true => {
                            if full_path.ends_with("/") {
                                format!("{}**/*", full_path)
                            } else {
                                format!("{}/**/*", full_path)
                            }
                        }
                        false => full_path,
                    };

                    gitignore_globs.push(full_path);
                }

                gitignore_globs.push(root.join(".gitignore").to_string_lossy().to_string());
                gitignore_globs.push(root.join(".git/**/*").to_string_lossy().to_string());
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
                        {
                            return false;
                        }
                    }

                    if f.is_file() {
                        kept += 1;
                        return true;
                    } else {
                        return false;
                    }
                })
                .map(|f| {
                    let filepath = f.to_string_lossy().to_string();

                    (filepath, meta_index)
                }),
        );

        meta_index += 1;

        info!("Kept {} files from {}", kept, entry);
        println!("Kept {} files from {}", kept, entry);
    }

    info!("{} config entries", config_entries.len());

    let new_ledger = config_entries
        .into_iter()
        .map(|s| LedgerEntry {
            filepath: s.0.clone(),
            hash: get_hash(&s.0).unwrap(),
            meta: config_ledger[s.1].meta.clone(),
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
                let mut meta_string = String::new();
                for (i, m) in entry.meta.iter().enumerate() {
                    if i < entry.meta.len() - 1 {
                        meta_string.push_str(&format!("{},", &m[2..]));
                    } else {
                        meta_string.push_str(&m[2..]);
                    }
                }

                meta_string = meta_string.trim().to_string();
                writeln!(file, "{} {} {}", entry.filepath, entry.hash, meta_string)?;
            }
        }
        Err(e) => {
            error!("Failed to write ledger file: {}", e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_common::*;
    use crate::write_file;

    #[test]
    fn read_ruleset_test() {
        let _cleanup = Cleanup;

        assert!(setup().is_ok());

        let rules = get_indexing_rules();
        assert!(rules.is_ok());

        let rules = rules.unwrap();

        // should really check the values themselves here but i'm too lazy to think through a
        // framework for doing so
        assert!(rules.contains_key("*"));
        assert!(rules.get("*").unwrap().len() == 3);

        assert!(rules.contains_key("rs"));
        assert!(rules.get("rs").unwrap().len() == 1);

        assert!(rules.contains_key("md"));
        assert!(rules.get("md").unwrap().len() == 1);
    }

    #[test]
    fn read_ledger_test() {
        let _cleanup = Cleanup;

        assert!(setup().is_ok());
        assert!(sync_ledger_config().is_ok());

        let entries = read_ledger();
        assert!(entries.is_ok());

        let tracked_files = get_tracked_files();
        let entries = entries.unwrap();

        for entry in entries.iter() {
            println!("looking for filepath {}", entry.filepath);
            assert!(tracked_files.iter().any(|tf| entry.filepath.contains(tf)));

            let mut meta_set = get_meta()
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<String>>();

            for m in entry.meta.iter() {
                println!("looking for meta {}", m);
                assert!(meta_set.remove(m));
            }
        }
    }

    // checking whether the tracked files are added to a fresh ledger
    // and unchecked files are not added
    #[test]
    fn sync_ledger_config_new() {
        let _cleanup = Cleanup;

        assert!(setup().is_ok());
        assert!(sync_ledger_config().is_ok());

        let ledger_path = crate::config::get_local_dir().join("ledger");

        let ledger_result = std::fs::read_to_string(ledger_path);
        assert!(ledger_result.is_ok());

        let contents = ledger_result.unwrap();
        let lines = contents
            .split("\n")
            .filter(|l| l.len() > 0)
            .collect::<Vec<&str>>();

        println!("local ledger contents:\n{}", contents);

        let tracked_files = get_tracked_files();
        assert_eq!(lines.len(), tracked_files.len());

        for line in lines {
            let items = line.split(" ").collect::<Vec<&str>>();
            assert_eq!(items.len(), 3);

            assert!(tracked_files.iter().any(|f| items[0].contains(f)));
        }
    }

    // tests updating an existing ledger to include new files
    #[test]
    fn sync_ledger_config_update_new() {
        let _cleanup = Cleanup;

        assert!(setup().is_ok());
        assert!(sync_ledger_config().is_ok());

        let new_files = vec![
            crate::config::get_home_dir()
                .join("test_repo")
                .join("new_rs.rs"),
            crate::config::get_home_dir()
                .join("test_repo")
                .join("no_extension"),
            crate::config::get_home_dir() // untracked .md file
                .join("test_repo")
                .join("new_md.md"),
        ];

        for nf in new_files.iter() {
            write_file!(nf, "testing");
            println!("wrote file {}", nf.to_str().unwrap());
        }

        assert!(sync_ledger_config().is_ok());

        let ledger_path = crate::config::get_local_dir().join("ledger");

        let ledger_result = std::fs::read_to_string(ledger_path);
        assert!(ledger_result.is_ok());

        let contents = ledger_result.unwrap();
        let lines = contents
            .split("\n")
            .filter(|l| l.len() > 0)
            .collect::<Vec<&str>>();

        println!("local ledger contents:\n{}", contents);

        let mut tracked_files = new_files
            .iter()
            .map(|nf| nf.to_string_lossy().to_string())
            .filter(|nf| !nf.ends_with(".md"))
            .collect::<Vec<String>>();

        tracked_files.extend(get_tracked_files());
        assert_eq!(lines.len(), tracked_files.len());

        for line in lines {
            let items = line.split(" ").collect::<Vec<&str>>();
            assert_eq!(items.len(), 3);

            assert!(tracked_files.iter().any(|f| items[0].contains(f)));
        }
    }
}
