use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

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
    pub fn from_file(filename: &str, _block: u64) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::open(filename)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let (embed_block, _) = EmbeddingBlock::from_bytes(&bytes, 0)?;

        Ok(embed_block)
    }

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
            directory.push((e.id, i));
        }
    }

    let directory = directory
        .into_iter()
        .map(|d| format!("{} {}", d.0, d.1))
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

    let mut directory = Vec::new();
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
        directory.push((current, i as u64));
        visited.insert(current);

        let mut neighbors = full_graph.get(&current).unwrap().clone();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (neighbor, _) in neighbors {
            if !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    let mut cache = EmbeddingCache::new(10 * BLOCK_SIZE as u32);

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

    let directory = directory
        .into_iter()
        .map(|d| format!("{} {}", d.0, d.1))
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

// filenames should be formatted `/whatever/directories/.../block_number`
// where `block_number` is a u64
pub fn read_blocks(filenames: &Vec<String>) -> Result<Vec<Box<Embedding>>, std::io::Error> {
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

        let block = EmbeddingBlock::from_file(filename, block_number)?;
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
        let block = match EmbeddingBlock::from_file(&filename.clone(), block_number) {
            Ok(block) => block,
            Err(e) => {
                eprintln!("Error reading embedding block {}: {}", block_number, e);
                return Err(e);
            }
        };

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
pub fn get_directory() -> Result<HashMap<u32, u64>, std::io::Error> {
    let data_dir = get_data_dir();
    let directory = std::fs::read_to_string(format!("{}/directory", data_dir.to_str().unwrap()))?;
    let directory = directory
        .split("\n")
        .map(|d| {
            let mut parts = d.split(" ");
            let id = parts.next().unwrap().parse::<u32>().unwrap();
            let block = parts.next().unwrap().parse::<u64>().unwrap();
            (id, block)
        })
        .collect::<HashMap<_, _>>();

    Ok(directory)
}
