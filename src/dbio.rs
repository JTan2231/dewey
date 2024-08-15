use crate::openai;
use std::io::{Read, Write};

use crate::info;
use crate::logger::Logger;

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
    pub embeddings: Vec<openai::Embedding>,
}

impl EmbeddingBlock {
    pub fn from_file(filename: &str) -> Result<Self, std::io::Error> {
        let block = match filename.parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Malformed filename, expected u64",
                ))
            }
        };

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
            let embedding = openai::Embedding::from_bytes(embedding_bytes);
            embeddings.push(embedding);
        }

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

        info!("{:?}", embedding_bytes);

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
pub fn sync_index() -> Result<(), std::io::Error> {
    let stale_files = crate::ledger::get_stale_files();
    let stale_sources = stale_files
        .iter()
        .map(|f| openai::EmbeddingSource {
            filepath: f.clone(),
            subset: None,
        })
        .collect::<Vec<_>>();

    let embeddings = openai::embed(&stale_sources)?;
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

pub fn test_block(block: u64) -> Result<(), std::io::Error> {
    let filename = format!("{}", block);
    let embedding_block = EmbeddingBlock::from_file(&filename)?;
    for embedding in embedding_block.embeddings.iter() {
        info!("{:?}", embedding.source_file);
    }

    info!("{:?}", embedding_block.embeddings[0]);

    Ok(())
}
