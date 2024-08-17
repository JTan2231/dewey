use std::io::{Read, Write};

use crate::config::get_data_dir;
use crate::hnsw::normalize;
use crate::info;
use crate::logger::Logger;
use crate::openai::{embed, Embedding, EmbeddingSource};

struct IndexedItem {
    filename: String,
    block: u64,
    offset: u64,
}

impl IndexedItem {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.filename.len().to_be_bytes());
        bytes.extend_from_slice(&self.filename.as_bytes());
        bytes.extend_from_slice(&self.block.to_be_bytes());
        bytes.extend_from_slice(&self.offset.to_be_bytes());

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let filename_len = u64::from_be_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let filename = String::from_utf8(bytes[8..8 + filename_len].to_vec()).unwrap();
        let block = u64::from_be_bytes(
            bytes[8 + filename_len..16 + filename_len]
                .try_into()
                .unwrap(),
        );
        let offset = u64::from_be_bytes(
            bytes[16 + filename_len..24 + filename_len]
                .try_into()
                .unwrap(),
        );

        Self {
            filename,
            block,
            offset,
        }
    }
}

// these have a limit of 1024 embeddings each
// format is:
// - 8 bytes: number of contained embeddings
// - n bytes: offset of each embedding
//
// followed by a contiguous series of the embeddings:
// - 8 bytes: size of EmbeddingSource
// - n bytes: EmbeddingSource
// - 8 bytes: size of Embedding
// - n bytes: Embedding
pub struct EmbeddingBlock {
    block: u64,
    pub embeddings: Vec<Embedding>,
}

impl EmbeddingBlock {
    pub fn from_file(filename: &str, block: u64) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::open(filename)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let u64_size = std::mem::size_of::<u64>();

        let num_embeddings = u64::from_be_bytes(bytes[0..u64_size].try_into().unwrap()) as usize;
        let mut offsets = Vec::new();
        for i in 0..num_embeddings {
            let offset = u64::from_be_bytes(
                bytes[u64_size + i * u64_size..u64_size + (i + 1) * u64_size]
                    .try_into()
                    .unwrap(),
            );
            offsets.push(offset);
        }

        let mut embeddings = Vec::new();
        for i in 0..num_embeddings {
            let start = offsets[i] as usize;
            let end = if i == num_embeddings - 1 {
                bytes.len()
            } else {
                offsets[i + 1] as usize
            };

            let embedding_bytes = &bytes[start..end];
            let embedding = Embedding::from_bytes(embedding_bytes);
            embeddings.push(embedding);
        }

        info!("loaded block {} from {}", block, filename);

        Ok(Self { block, embeddings })
    }

    fn to_file(&self, filename: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filename)?;

        let u64_size = std::mem::size_of::<u64>();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.embeddings.len() as u64).to_be_bytes());

        let embedding_bytes = self
            .embeddings
            .iter()
            .map(|e| e.to_bytes())
            .collect::<Vec<_>>();

        let mut offsets = Vec::new();
        let mut offset = u64_size + self.embeddings.len() * u64_size;
        for embedding_bytes in embedding_bytes.iter() {
            offsets.push(offset);
            offset += embedding_bytes.len();
        }

        for offset in offsets {
            bytes.extend_from_slice(&offset.to_be_bytes());
        }

        for embedding_bytes in embedding_bytes.iter() {
            bytes.extend_from_slice(&embedding_bytes);
        }

        info!("Writing {} bytes to {}", bytes.len(), filename);
        file.write_all(&bytes)?;

        Ok(())
    }
}

fn read_index() -> Result<Vec<IndexedItem>, std::io::Error> {
    let items = Vec::new();

    Ok(items)
}

// synchronizes the index with the current ledger
// TODO: ledgers need to include subsets of files
//       we also need a proper tokenizer
pub fn sync_index(full_embed: bool) -> Result<(), std::io::Error> {
    let stale_sources = match full_embed {
        true => crate::ledger::read_ledger()
            .into_iter()
            .map(|le| EmbeddingSource {
                filepath: le.filepath.clone(),
                subset: None,
            })
            .collect::<Vec<_>>(),
        false => {
            let stale_files = crate::ledger::get_stale_files()?;
            stale_files
                .iter()
                .map(|f| EmbeddingSource {
                    filepath: f.clone(),
                    subset: None,
                })
                .collect::<Vec<_>>()
        }
    };

    let embeddings = embed(&stale_sources)?;
    let blocks = embeddings.chunks(1024);
    for (i, block) in blocks.enumerate() {
        let filename = format!("{}", i);
        let embedding_block = EmbeddingBlock {
            block: i as u64,
            embeddings: block.to_vec(),
        };

        embedding_block.to_file(&filename)?;
    }

    Ok(())
}

pub fn get_blocks() -> Result<Vec<Box<Embedding>>, std::io::Error> {
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

    let mut embeddings = Vec::new();
    for block_number in block_numbers {
        let block = match EmbeddingBlock::from_file(
            &format!("{}/{}", data_dir.to_str().unwrap(), block_number),
            block_number,
        ) {
            Ok(block) => block,
            Err(e) => {
                eprintln!("Error reading embedding block {}: {}", block_number, e);
                return Err(e);
            }
        };

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
