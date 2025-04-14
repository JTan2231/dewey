use chamber_common::Logger;
use chamber_common::{error, get_data_dir, info, lprint};

use crate::cache::EmbeddingCache;
use crate::dbio::BLOCK_SIZE;
use crate::hnsw::{Filter, Query, HNSW};
pub use crate::openai::{embed, EmbeddingSource};

mod cache;
pub mod config;
pub mod dbio;
pub mod hnsw;
pub mod ledger;
mod openai;
mod parsing;
pub mod serialization;
pub mod test_common;

pub struct Dewey {
    index: hnsw::HNSW,
    cache: EmbeddingCache,
}

impl Dewey {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        crate::config::setup()?;

        lprint!(info, "Dewey: Verifying OpenAI API key...");
        let key = std::env::var("OPENAI_API_KEY").map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("OpenAI API key not found: {}", e),
            )
        })?;

        if key.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "OpenAI API key is empty",
            )));
        }

        // We're rebuilding the index from the blocks for now because it's assumed that the number
        // of messages will be small enough to warrant this
        // More than a few blocks, however. will probably warrant some sort of process for building
        // these in the background
        //
        // TODO: Figure something out to keep the index fresh without compromising performance
        Ok(Self {
            index: HNSW::new(true)?,
            cache: EmbeddingCache::new((20 * BLOCK_SIZE) as u32)?,
        })
    }

    // TODO: better define how filters should be passed
    pub fn query(
        &mut self,
        query_filepath: &str,
        filters: Vec<String>,
        k: usize,
    ) -> Result<Vec<EmbeddingSource>, std::io::Error> {
        let embedding = match embed(&EmbeddingSource {
            filepath: query_filepath.to_string(),
            meta: std::collections::HashSet::new(),
            subset: None,
        }) {
            Ok(e) => e,
            Err(e) => {
                error!("Failed to create embedding: {}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
            }
        };

        let filters = filters
            .iter()
            .map(|f| Filter::from_string(&f.to_string()).unwrap())
            .collect::<Vec<Filter>>();

        let query = Query { embedding, filters };

        Ok(self
            .index
            .query(&mut self.cache, &query, k, 200)
            .iter()
            .map(|p| p.0.source_file.clone())
            .collect())
    }

    // This returns an empty json object {} on success
    // or an object with just an `error` key on error
    pub fn reindex(&mut self, filepath: String) -> Result<(), std::io::Error> {
        crate::dbio::update_file_embeddings(&filepath, &mut self.index)
    }

    /// Add a new embedding to the system from the given file
    ///
    /// This updates both:
    /// - The embedding store in the OS file system
    /// - The in-memory HNSW index
    ///
    /// Alongside related metadata + other housekeeping files in the OS filesystem:
    /// - Embedding store directory
    /// - HNSW index file
    pub fn add_embedding(&mut self, filepath: String) -> Result<(), std::io::Error> {
        let mut embedding = embed(&EmbeddingSource {
            filepath,
            subset: None,
            meta: std::collections::HashSet::new(),
        })?;

        // TODO: ledger integration here at some point
        //       from what I understand the ledger is only for syncing
        //       between the local file system and the embedding store
        //       since William is adding things to the store directly,
        //       it can bypass the ledger
        //       but it would be nice to have file/embedding syncing
        //       and tracking all taking place in one spot (the ledger)

        match dbio::add_new_embedding(&mut embedding) {
            Ok(_) => {}
            Err(e) => {
                error!("error adding embedding to store: {}", e);
                return Err(e);
            }
        };

        lprint!(info, "Created embedding with id: {}", embedding.id);
        lprint!(info, "Finished writing embedding to file system");

        self.cache.refresh_directory()?;
        lprint!(info, "Refreshed cache directory");

        match self.index.insert(&mut self.cache, &embedding) {
            Ok(_) => {}
            Err(e) => {
                error!("Error adding embedding to index: {}", e);
                return Err(e);
            }
        };

        match self
            .index
            .serialize(&get_data_dir().join("index").to_str().unwrap().to_string())
        {
            Ok(_) => {}
            Err(e) => {
                error!("error serializing index: {}", e);
                return Err(e);
            }
        };

        lprint!(info, "Updated index with new embedding");

        Ok(())
    }
}
