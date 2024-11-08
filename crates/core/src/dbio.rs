use std::collections::{HashMap, HashSet};
use std::io::Write;

use serialize_macros::Serialize;

use crate::cache::EmbeddingCache;
use crate::config::get_data_dir;
use crate::hnsw::{normalize, HNSW};
use crate::logger::Logger;
use crate::openai::{embed_bulk, Embedding, EmbeddingSource};
use crate::serialization::Serialize;
use crate::{error, info};

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

// synchronizes the index with the current ledger
// TODO: ledgers need to include subsets of files
//       we also need a proper tokenizer
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

    for (i, e) in embeddings.iter_mut().enumerate() {
        e.id = i as u64;
    }

    let mut directory = Vec::new();

    let data_dir = get_data_dir();

    let existing_blocks = std::fs::read_dir(data_dir.clone())?;
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
        let filename = format!("{}/{}", data_dir.to_str().unwrap(), i);
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

    let full_graph = index.get_last_layer();

    let mut blocks = vec![Vec::new()];
    let mut i = 0;

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    stack.push(*full_graph.iter().nth(0).unwrap().0);

    while let Some(current) = stack.pop() {
        if visited.contains(&current) {
            continue;
        }

        if visited.len() % (full_graph.len() / 10) == 0 {
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
    let data_dir = get_data_dir();
    let temp_dir = format!("{}/temp", data_dir.to_str().unwrap());

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

    for entry in std::fs::read_dir(data_dir.clone())? {
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

    std::fs::remove_file(format!("{}/directory", data_dir.to_str().unwrap()))?;

    for entry in std::fs::read_dir(temp_dir.clone())? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name() {
                if let Some(filename) = filename.to_str() {
                    if filename.parse::<u64>().is_ok() {
                        std::fs::rename(
                            path.clone(),
                            format!("{}/{}", data_dir.to_str().unwrap(), filename),
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

// filenames should be formatted `/whatever/directories/.../block_number`
// where `block_number` is a u64
pub fn read_embedding_blocks(
    filenames: &Vec<String>,
) -> Result<Vec<Box<Embedding>>, std::io::Error> {
    let mut embeddings = Vec::new();
    for filename in filenames {
        let block_number = match filename.split("/").last().unwrap().parse::<u64>() {
            Ok(block_number) => block_number,
            Err(e) => {
                eprintln!(
                    "Error parsing block number from filename {}: {}",
                    filename, e
                );
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid block number",
                ));
            }
        };

        let block = read_embedding_block(block_number)?;
        embeddings.extend(
            block
                .embeddings
                .into_iter()
                .map(|mut embedding| {
                    normalize(&mut embedding);
                    Box::new(embedding)
                })
                .collect::<Vec<_>>(),
        );
    }

    Ok(embeddings)
}

pub fn read_embedding_block(block_number: u64) -> Result<EmbeddingBlock, std::io::Error> {
    let data_dir = get_data_dir();

    let bytes = match std::fs::read(&format!("{}/{}", data_dir.to_str().unwrap(), block_number)) {
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
    let data_dir = get_data_dir();
    let mut block_numbers = Vec::new();
    for entry in std::fs::read_dir(data_dir.clone())? {
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
        let filename = format!("{}/{}", data_dir.to_str().unwrap(), block_number);
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
pub fn get_directory() -> Result<Directory, std::io::Error> {
    let data_dir = get_data_dir();
    let directory = std::fs::read_to_string(format!("{}/directory", data_dir.to_str().unwrap()))?;
    let directory = directory
        .split("\n")
        .map(|d| {
            let parts = d.split(" ").collect::<Vec<&str>>();
            let id = parts[0].parse::<u32>().unwrap();
            let filepath = parts[1..parts.len() - 1].join("");
            let block = parts[parts.len() - 1].parse::<u64>().unwrap();

            (id, filepath, block)
        })
        .collect::<Vec<_>>();

    let mut id_map = HashMap::new();
    let mut file_map = HashMap::new();
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
pub fn update_file_embeddings(filepath: &str, index: &mut HNSW) -> Result<(), std::io::Error> {
    let directory = match get_directory() {
        Ok(d) => d,
        Err(e) => {
            error!("error reading directory: {}", e);
            return Err(e);
        }
    };

    let id_start = directory.id_map.len() as u64 + 1;

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

    for (i, e) in new_embeddings.iter_mut().enumerate() {
        e.id = id_start + i as u64;
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
