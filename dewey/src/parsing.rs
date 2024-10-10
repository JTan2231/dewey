use std::io::Read;

use crate::ledger::{get_indexing_rules, IndexRuleType};
use crate::openai::EmbeddingSource;

use crate::logger::Logger;
use crate::{error, info};

pub fn read_source(source: &EmbeddingSource) -> Result<String, std::io::Error> {
    let mut file = match std::fs::File::open(&source.filepath) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open file: {:?}", e);
            return Err(e);
        }
    };

    let mut buffer = Vec::new();
    let contents = match file.read_to_end(&mut buffer) {
        Ok(_) => String::from_utf8_lossy(&buffer).to_string(),
        Err(e) => {
            error!("Failed to read from file {}: {:?}", source.filepath, e);
            return Err(e);
        }
    };

    // TODO: we can easily get away without loading the entire file into memory
    let contents = match source.subset {
        Some((start, end)) => contents[start as usize..end as usize].to_string(),
        _ => contents,
    };

    // CRLF -> LF
    let contents = contents.replace("\r\n", "\n");

    Ok(contents)
}

// TODO: a proper tokenizer
pub const TOKEN_LIMIT: usize = 8192;
fn separator_split(
    source: &EmbeddingSource,
    separator: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let contents = read_source(&source)?;
    let chars = contents.chars().collect::<Vec<char>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() - separator.len() {
        let window = String::from_iter(&chars[i..i + separator.len()]);
        if window == *separator || chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();

            i += separator.len();
        } else {
            let c = chars[i].to_string();
            chunk.push_str(&c);
            i += c.len();
        }
    }

    if !chunk.is_empty() {
        chunks.push((
            chunk.clone(),
            (contents.len() - chunk.len(), contents.len()),
        ));
    }

    Ok(chunks)
}

// this only has a _separator argument so it can be used as a function pointer
// for either this or `separator_split`
fn naive_split(
    source: &EmbeddingSource,
    _separator: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let source_contents = read_source(&source)?;
    let chars = source_contents.chars().collect::<Vec<_>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();
            i += 1;
        } else {
            let c = chars[i].to_string();
            chunk.push_str(&c);
            i += c.len();
        }
    }

    if !chunk.is_empty() {
        chunks.push((
            chunk.clone(),
            (source_contents.len() - chunk.len(), source_contents.len()),
        ));
    }

    Ok(chunks)
}

fn max_length_split(
    source: &EmbeddingSource,
    max_length: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let source_contents = read_source(&source)?;
    let chars = source_contents.chars().collect::<Vec<_>>();
    let max_length = max_length.parse::<usize>().unwrap();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chunk.len() >= TOKEN_LIMIT || chunk.len() >= max_length {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();
            i += 1;
        } else {
            let c = chars[i].to_string();
            chunk.push_str(&c);
            i += c.len();
        }
    }

    if !chunk.is_empty() {
        chunks.push((
            chunk.clone(),
            (source_contents.len() - chunk.len(), source_contents.len()),
        ));
    }

    Ok(chunks)
}

struct FunctionDefinition {
    pub definition: String,
    pub name: String,
    pub begin: usize,
    pub end: usize,
}

#[allow(unused_assignments)]
fn function_split(
    source: &EmbeddingSource,
    _max_length: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let filepath = std::path::PathBuf::from(&source.filepath);
    let mut language_fn = None;
    let mut language = "";
    match filepath.extension() {
        Some(ext) => match ext.to_str() {
            Some("rs") => {
                language = "rust";
                language_fn = Some(tree_sitter_rust::language());
            }
            Some("py") => {
                language = "python";
                language_fn = Some(tree_sitter_python::language());
            }
            Some("js") => {
                language = "javascript";
                language_fn = Some(tree_sitter_javascript::language());
            }
            _ => {
                error!(
                    "Unsupported file extension {}, using a naive split instead",
                    ext.to_str().unwrap_or("_empty")
                );
                return naive_split(source, _max_length);
            }
        },
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing file extension",
            ));
        }
    }

    let language_fn = language_fn.unwrap();

    // TODO: it's probably pretty stupid to initialize
    //       a separate parser for each call
    //       i'd imagine there's a much smarter way to go about this
    let mut parser = tree_sitter::Parser::new();
    match parser.set_language(&language_fn) {
        Ok(_) => {}
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to set language: {}", e),
            ));
        }
    };

    let query = match language {
        "rust" => tree_sitter::Query::new(
            &language_fn,
            r#"
            (function_item
                name: (identifier) @func_name
                parameters: (parameters) @func_params
                return_type: (type_identifier)? @return_type
                body: (block)? @func_body
            ) @func_def

            (function_signature_item
                name: (identifier) @func_name
                parameters: (parameters) @func_params
                return_type: (type_identifier)? @return_type
            ) @func_def
            "#,
        )
        .unwrap(),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported language: {}", language),
            ))
        }
    };

    let contents = match read_source(source) {
        Ok(contents) => contents,
        Err(e) => {
            error!("error reading file {}", source.filepath);
            return Err(e);
        }
    };

    let tree = parser
        .parse(&contents, None)
        .expect("failed to parse source");
    let mut query_cursor = tree_sitter::QueryCursor::new();
    let matches = query_cursor.matches(&query, tree.root_node(), contents.as_bytes());

    let mut definitions = Vec::new();
    for match_ in matches {
        let mut definition = FunctionDefinition {
            definition: String::new(),
            name: String::new(),
            begin: 0,
            end: 0,
        };

        for capture in match_.captures {
            let range = capture.node.byte_range();
            match query.capture_names()[capture.index as usize] {
                "func_def" => {
                    definition.definition = contents[range.clone()].to_string();
                    definition.begin = range.start;
                    definition.end = range.end;
                }
                "func_name" => definition.name = contents[range].to_string(),

                _ => {}
            }
        }

        definitions.push(definition);
    }

    let mut chunks = Vec::new();
    for definition in definitions {
        // if the function definition is too big for a single chunk,
        // we just run a naive split on it
        //
        // note that we add `definition.begin` to each index position
        // since they're to be in reference to the file start
        if definition.definition.len() >= TOKEN_LIMIT {
            let chars = definition.definition.chars().collect::<Vec<_>>();
            let mut chunk = String::new();
            let mut i = 0;
            while i < chars.len() {
                if chunk.len() >= TOKEN_LIMIT {
                    chunks.push((
                        chunk.clone(),
                        (i - chunk.len() + definition.begin, i + definition.begin),
                    ));
                    chunk.clear();
                    i += 1;
                } else {
                    let c = chars[i].to_string();
                    chunk.push_str(&c);
                    i += c.len();
                }
            }

            if !chunk.is_empty() {
                chunks.push((
                    chunk.clone(),
                    (
                        definition.definition.len() - chunk.len() + definition.begin,
                        definition.definition.len() + definition.begin,
                    ),
                ));
            }
        } else {
            chunks.push((definition.definition, (definition.begin, definition.end)));
        }
    }

    Ok(chunks)
}

// NOTE: _only_ supports ascii
pub fn batch_sources(
    sources: &Vec<EmbeddingSource>,
) -> Result<Vec<Vec<(EmbeddingSource, String)>>, std::io::Error> {
    let indexing_rules = get_indexing_rules()?;
    let base = Vec::new();
    let global_rules = indexing_rules.get("*").unwrap_or(&base);
    info!("batching with rules: {:?}", indexing_rules);
    // API requests need batched up to keep from exceeding token limits
    let mut batches: Vec<Vec<(EmbeddingSource, String)>> = vec![Vec::new()];
    for source in sources {
        let extension = match source.filepath.split(".").last() {
            Some(extension) => extension,
            _ => {
                error!("Failed to get extension from file: {:?}", source.filepath);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Failed to get extension from file",
                ));
            }
        };

        let mut rules = global_rules.clone();
        if let Some(extension_rules) = indexing_rules.get(extension) {
            rules.extend(extension_rules.clone());
        }

        let mut rule_arg = "".to_string();
        let split_function: fn(
            &EmbeddingSource,
            &String,
        ) -> Result<Vec<(String, (usize, usize))>, std::io::Error> = {
            let mut rule_type = "".to_string();
            for rule in rules.iter() {
                match rule.rule_type {
                    IndexRuleType::Split => {
                        rule_arg = rule.value.clone();
                        rule_type = "separator".to_string();
                    }
                    IndexRuleType::MaxLength => {
                        rule_arg = rule.value.clone();
                        rule_type = "max_length".to_string();
                    }
                    IndexRuleType::Code => {
                        rule_type = "code".to_string();
                    }
                    _ => (),
                }
            }

            match rule_type.as_str() {
                "separator" => separator_split,
                "max_length" => max_length_split,
                "code" => function_split,
                _ => naive_split,
            }
        };

        let mut contents_split = split_function(&source, &rule_arg)?;

        // there's probably a better way to apply these filters
        // in conjunction with the splitters
        for rule in rules {
            match rule.rule_type {
                IndexRuleType::MinLength => {
                    let min_length = rule.value.parse::<usize>().unwrap();
                    contents_split.retain(|(_, range)| range.1 - range.0 >= min_length);
                }
                IndexRuleType::Alphanumeric => {
                    contents_split.retain(|(contents, _)| {
                        contents
                            .chars()
                            .any(|c| c.is_alphanumeric() || c.is_whitespace())
                    });
                }
                _ => (),
            }
        }

        let mut split = batches.last_mut().unwrap();
        let mut split_len = 0;
        for (contents, window) in contents_split {
            if contents.len() + split_len >= TOKEN_LIMIT {
                batches.push(Vec::new());

                split = batches.last_mut().unwrap();
                split_len = 0;
            }

            if contents.len() > 0 {
                split_len += contents.len();
                let new_source = EmbeddingSource {
                    filepath: source.filepath.clone(),
                    meta: source.meta.clone(),
                    subset: Some((window.0 as u64, window.1 as u64)),
                };
                split.push((new_source, contents));
            }
        }
    }

    batches.retain(|batch| !batch.is_empty());

    info!(
        "batched {} sources into {} batches",
        sources.len(),
        batches.len()
    );

    println!(
        "batched {} sources into {} batches",
        sources.len(),
        batches.len()
    );

    Ok(batches)
}
