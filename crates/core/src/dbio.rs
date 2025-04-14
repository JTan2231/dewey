use std::collections::{HashMap, HashSet};
use std::io::Write;

use chamber_common::Logger;
use chamber_common::{error, get_data_dir, get_local_dir, info, lprint};
use serialize_macros::Serialize;

use crate::cache::EmbeddingCache;
use crate::hnsw::{normalize, HNSW};
use crate::openai::{embed_bulk, Embedding, EmbeddingSource};
use crate::serialization::Serialize;

// TODO: this could probably be a config parameter
pub const BLOCK_SIZE: usize = 1024;

#[derive(Serialize)]
pub struct EmbeddingBlock {
    block: u64,
    pub embeddings: Vec<Embedding>,
}

impl EmbeddingBlock {
    fn to_file(&self, filename: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filename)?;

        let bytes = self.to_bytes();
        info!("Writing {} bytes to {}", bytes.len(), filename);
        file.write_all(&bytes)?;

        Ok(())
    }
}

struct DirectoryEntry {
    id: u32,
    filepath: String,
}

// TODO: more clarification needed
//
// directory for which embeddings are in which blocks
pub struct Directory {
    pub file_map: HashMap<String, u64>,
    pub id_map: HashMap<u32, u64>,
    pub file_id_map: HashMap<String, u32>,
}

impl Directory {
    pub fn len(&self) -> usize {
        self.id_map.len()
    }
}

fn write_directory(entries: &Vec<(DirectoryEntry, u32)>) -> Result<(), std::io::Error> {
    let directory = entries
        .into_iter()
        .map(|d| format!("{} {} {}", d.0.id, d.0.filepath, d.1))
        .collect::<Vec<_>>();
    let count = directory.len();
    let directory = directory.join("\n");

    std::fs::write(
        format!("{}/directory", get_data_dir().to_str().unwrap()),
        directory,
    )?;

    info!("Wrote directory with {} entries", count);

    Ok(())
}

// NOTE: not thread safe
fn get_next_id() -> Result<u64, std::io::Error> {
    let counter_path = get_local_dir().join("id_counter");
    let contents = match std::fs::read_to_string(&counter_path) {
        Ok(c) => {
            if c.is_empty() {
                "0".to_string()
            } else {
                c
            }
        }
        Err(e) => {
            error!("error opening ID counter file: {e}");
            return Err(e);
        }
    };

    let last_id = match contents.parse::<u64>() {
        Ok(id) => id,
        Err(e) => {
            error!("error reading ID counter file: {e}");
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
        }
    };

    let new_id = last_id + 1;

    if let Err(e) = std::fs::write(&counter_path, new_id.to_string()) {
        error!("error writing new ID: {e}");
        return Err(e);
    }

    Ok(new_id)
}

// synchronizes the index with the current ledger
// TODO: ledgers need to include subsets of files
//       we also need a proper tokenizer
//
// TODO: we need to implement handling for embedding DBs
//       that don't fit in memory
//       and this needs to happen project-wide
//
// TODO: decide how much this is really needed
//       right now it's not being used
pub fn sync_index(full_embed: bool) -> Result<(), std::io::Error> {
    let stale_sources = match full_embed {
        true => crate::ledger::read_ledger()?
            .into_iter()
            .map(|entry| EmbeddingSource {
                filepath: entry.filepath.clone(),
                meta: entry.meta.clone(),
                subset: None,
            })
            .collect::<Vec<_>>(),
        false => {
            let stale_files = crate::ledger::get_stale_files()?;
            stale_files
                .iter()
                .map(|entry| EmbeddingSource {
                    filepath: entry.filepath.clone(),
                    meta: entry.meta.clone(),
                    subset: None,
                })
                .collect::<Vec<_>>()
        }
    };

    let mut embeddings = embed_bulk(&stale_sources)?;

    for e in embeddings.iter_mut() {
        e.id = get_next_id()?;
    }

    let mut directory = Vec::new();

    // TODO: there definitely need to be some better guarantees here
    let existing_blocks = std::fs::read_dir(get_data_dir().clone())?;
    for entry in existing_blocks {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename) = filename.to_str() {
                    if filename.parse::<u64>().is_ok() {
                        std::fs::remove_file(path)?;
                    }
                }
            }
        }
    }

    let blocks = embeddings.chunks(BLOCK_SIZE);
    for (i, block) in blocks.enumerate() {
        let filename = format!("{}/{}", get_data_dir().to_str().unwrap(), i);
        let embedding_block = EmbeddingBlock {
            block: i as u64,
            embeddings: block.to_vec(),
        };

        embedding_block.to_file(&filename)?;

        for e in block {
            directory.push((
                DirectoryEntry {
                    id: e.id as u32,
                    filepath: e.source_file.filepath.clone(),
                },
                i as u32,
            ));
        }
    }

    // TODO: need some sort of follow-up to handle unfinished business regarding the directory
    match write_directory(&directory) {
        Ok(_) => {}
        Err(e) => {
            error!("error writing directory: {}", e);
            return Err(e);
        }
    };

    Ok(())
}

// optimizes embedding placement in blocks based on their distance from their neighbors
// also syncs meta changes from the ledger
//
// TODO: needs refactored to fit william integration
pub fn reblock() -> Result<(), std::io::Error> {
    let index = match HNSW::new(false) {
        Ok(index) => index,
        Err(e) => {
            eprintln!("Error creating index: {}", e);
            eprintln!("Note: this operation requires an index to be present");
            eprintln!("Run `hnsw -s` to recreate your index");
            return Err(e);
        }
    };

    let full_graph = match index.get_last_layer() {
        Some(g) => g,
        None => {
            info!("index is empty; nothing to do.");
            return Ok(());
        }
    };

    let mut blocks = vec![Vec::new()];
    let mut i = 0;

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    stack.push(*full_graph.iter().nth(0).unwrap().0);

    while let Some(current) = stack.pop() {
        if visited.contains(&current) {
            continue;
        }

        if full_graph.len() > 10 && visited.len() % (full_graph.len() / 10) == 0 {
            info!("blocked {} nodes into {} blocks", visited.len(), i + 1);
        }

        if blocks[i].len() >= BLOCK_SIZE {
            blocks.push(Vec::new());
            i += 1;
        }

        blocks[i].push(current);
        visited.insert(current);

        let mut neighbors = full_graph.get(&current).unwrap().clone();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (neighbor, _) in neighbors {
            if !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    let mut cache = EmbeddingCache::new(10 * BLOCK_SIZE as u32)?;

    // update meta
    let ledger = crate::ledger::read_ledger()?;
    let mut ledger_map = std::collections::HashMap::new();
    for entry in ledger.iter() {
        ledger_map.insert(entry.filepath.clone(), entry.meta.clone());
    }

    // create a temp directory in $DATA_DIR to hold all the blocks
    let temp_dir = format!("{}/temp", get_data_dir().to_str().unwrap());

    if std::fs::metadata(&temp_dir).is_ok() {
        std::fs::remove_dir_all(&temp_dir)?;
    }

    std::fs::create_dir(&temp_dir)?;

    let mut directory = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        let filename = format!("{}/{}", temp_dir, i);
        let mut embeddings = Vec::new();
        for id in block {
            let mut embedding = cache.get(*id as u32).unwrap();
            embedding.source_file.meta = match ledger_map.get(&embedding.source_file.filepath) {
                Some(meta) => meta.clone(),
                None => {
                    error!(
                        "File {} unaccounted for in ledger! Ignoring meta",
                        embedding.source_file.filepath
                    );
                    embedding.source_file.meta
                }
            };

            directory.push((
                DirectoryEntry {
                    id: embedding.id as u32,
                    filepath: embedding.source_file.filepath.clone(),
                },
                i as u32,
            ));

            embeddings.push(*embedding);
        }

        let embedding_block = EmbeddingBlock {
            block: i as u64,
            embeddings,
        };

        embedding_block.to_file(&filename)?;
    }

    for entry in std::fs::read_dir(get_data_dir().clone())? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename) = filename.to_str() {
                    if filename.parse::<u64>().is_ok() {
                        std::fs::remove_file(path)?;
                    }
                }
            }
        }
    }

    std::fs::remove_file(format!("{}/directory", get_data_dir().to_str().unwrap()))?;

    for entry in std::fs::read_dir(temp_dir.clone())? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename) = filename.to_str() {
                    if filename.parse::<u64>().is_ok() {
                        std::fs::rename(
                            path.clone(),
                            format!("{}/{}", get_data_dir().to_str().unwrap(), filename),
                        )?;
                    }
                }
            }
        }
    }

    std::fs::remove_dir_all(&temp_dir)?;

    match write_directory(&directory) {
        Ok(_) => {}
        Err(e) => {
            error!("error writing directory: {}", e);
            return Err(e);
        }
    };

    Ok(())
}

pub fn read_embedding_block(block_number: u64) -> Result<EmbeddingBlock, std::io::Error> {
    let bytes = match std::fs::read(&format!(
        "{}/{}",
        get_data_dir().to_str().unwrap(),
        block_number
    )) {
        Ok(b) => b,
        Err(e) => {
            error!("error reading block file {}: {}", block_number, e);
            return Err(e);
        }
    };

    let (block, _) = match EmbeddingBlock::from_bytes(&bytes, 0) {
        Ok(b) => b,
        Err(e) => {
            error!("error parsing block file {}: {}", block_number, e);
            return Err(e);
        }
    };

    Ok(block)
}

pub struct BlockEmbedding {
    pub block_number: u64,
    pub embedding: Box<Embedding>,
    pub source_file: String,
}

// returns boxes of the embeddings and the block files from which they were read
pub fn get_all_blocks() -> Result<Vec<BlockEmbedding>, std::io::Error> {
    let mut block_numbers = Vec::new();
    for entry in std::fs::read_dir(get_data_dir().clone())? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename) = filename.to_str() {
                    if filename.parse::<u64>().is_ok() {
                        block_numbers.push(filename.parse::<u64>().unwrap());
                    }
                }
            }
        }
    }

    let mut block_embeddings = Vec::new();
    for block_number in block_numbers {
        let filename = format!("{}/{}", get_data_dir().to_str().unwrap(), block_number);
        let block = read_embedding_block(block_number)?;

        for be in block
            .embeddings
            .into_iter()
            .map(|mut embedding| {
                normalize(&mut embedding);
                Box::new(embedding)
            })
            .collect::<Vec<_>>()
        {
            block_embeddings.push(BlockEmbedding {
                block_number,
                embedding: be,
                source_file: filename.clone(),
            });
        }
    }

    Ok(block_embeddings)
}

// TODO: at what point should we worry about holding this whole thing in memory?
//       it shouldn't stay like this forever
//       i think the directory should be grouped in separate files by both:
//         - layers
//       and
//         - embedding blocks
pub fn get_directory() -> Result<Directory, std::io::Error> {
    let directory =
        match std::fs::read_to_string(format!("{}/directory", get_data_dir().to_str().unwrap())) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                error!("error reading directory file: {}", e);
                return Err(e);
            }
        };

    let directory = directory
        .split("\n")
        .filter(|l| !l.is_empty())
        .map(|d| {
            let parts = d.split(" ").collect::<Vec<&str>>();
            let id = parts[0].parse::<u32>().unwrap();
            let filepath = parts[1..parts.len() - 1].join("");
            let block = parts[parts.len() - 1].parse::<u64>().unwrap();

            (id, filepath, block)
        })
        .collect::<Vec<_>>();

    // Embedding ID -> block number
    let mut id_map = HashMap::new();

    // Embedding source filepath -> block number
    let mut file_map = HashMap::new();

    // Embedding source filepath -> embedding ID
    let mut file_id_map = HashMap::new();

    for entry in directory.iter() {
        id_map.insert(entry.0, entry.2);
        file_map.insert(entry.1.clone(), entry.2);
        file_id_map.insert(entry.1.clone(), entry.0);
    }

    Ok(Directory {
        id_map,
        file_map,
        file_id_map,
    })
}

// TODO: how does this affect indexing?
//       i think things need reindexed + reblocked after updates here
//
// updates the embeddings for the given file
// requires the file to have already been indexed
pub fn update_file_embeddings(filepath: &str, index: &mut HNSW) -> Result<(), std::io::Error> {
    let directory = match get_directory() {
        Ok(d) => d,
        Err(e) => {
            error!("error reading directory: {}", e);
            return Err(e);
        }
    };

    let target_block = match directory.file_map.get(filepath) {
        Some(b) => b,
        None => {
            error!(
                "filepath {} not catalogued in Directory, aborting update",
                filepath
            );
            return Ok(());
        }
    };

    let mut block = read_embedding_block(*target_block)?;

    let mut meta = HashSet::new();
    let mut to_delete = Vec::new();
    for e in block.embeddings.iter() {
        if e.source_file.filepath == filepath {
            meta = e.source_file.meta.clone();
            to_delete.push(e.id);
        }
    }

    block
        .embeddings
        .retain(|e| e.source_file.filepath != filepath);

    let mut new_embeddings = embed_bulk(&vec![EmbeddingSource {
        filepath: filepath.to_string(),
        meta,
        subset: None,
    }])?;

    for e in new_embeddings.iter_mut() {
        e.id = get_next_id()?;
    }

    block.embeddings.extend(new_embeddings);

    let block_path = format!("{}/{}", get_data_dir().to_str().unwrap(), target_block);
    block.to_file(&block_path)?;

    for node in to_delete {
        index.remove_node(node);
    }

    index.serialize(&get_data_dir().join("index").to_str().unwrap().to_string())?;

    Ok(())
}

/// this adds a new embedding to the embedding store
///
/// the last block is chosen (arbitrarily) as its new home
/// the directory file is also updated with an entry for the new embedding
///
/// this _does not_ affect the HNSW index--in-memory or otherwise
/// updates to the index should take place with that struct directly
/// this function here is specifically for adding the embeddings
/// to the file system
pub fn add_new_embedding(embedding: &mut Embedding) -> Result<(), std::io::Error> {
    let last_block_number = match std::fs::read_dir(get_data_dir())
        .unwrap()
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let filename = entry.file_name();
            let filename_str = filename.to_str()?;

            // Try to parse the filename as a number
            filename_str.parse::<u64>().ok()
        })
        .max()
    {
        Some(bn) => bn,
        None => 0,
    };

    let mut block = match read_embedding_block(last_block_number) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => EmbeddingBlock {
            block: 0,
            embeddings: Vec::new(),
        },
        Err(e) => {
            return Err(e);
        }
    };

    embedding.id = get_next_id()?;
    block.embeddings.push(embedding.clone());

    let filepath = format!("{}/{}", get_data_dir().to_str().unwrap(), block.block);
    block.to_file(&filepath)?;

    lprint!(info, "Saved embedding to {}", filepath);

    let mut directory = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(get_data_dir().join("directory"))?;

    writeln!(
        directory,
        "\n{} {} {}",
        embedding.id, embedding.source_file.filepath, last_block_number
    )?;

    lprint!(info, "Directory updated");

    Ok(())
}
