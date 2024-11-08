use crate::hnsw::{Filter, Query, HNSW};
use crate::logger::Logger;
use crate::message::{DeweyResponse, DeweyResponseItem, RequestPayload};
use crate::openai::{embed, EmbeddingSource};

pub mod cache;
pub mod config;
pub mod dbio;
pub mod hnsw;
pub mod ledger;
pub mod logger;
pub mod message;
pub mod openai;
pub mod parsing;
pub mod serialization;
pub mod test_common;

// all server operations should go through this arc-mutexed state
// this is needed for thread safety with the addition of db-altering operations
pub struct ServerState {
    index: hnsw::HNSW,
}

impl ServerState {
    pub fn new() -> Result<Self, std::io::Error> {
        Ok(Self {
            index: HNSW::new(false)?,
        })
    }

    pub fn query(&self, payload: RequestPayload) -> Result<String, std::io::Error> {
        let (query, filters, k) = match payload {
            RequestPayload::Query { query, filters, k } => (query, filters, k),
            _ => {
                error!("malformed query request: {:?}", payload);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "malformed query request",
                ));
            }
        };

        let timestamp = chrono::Utc::now().timestamp_micros();
        let path = config::get_local_dir()
            .join("queries")
            .join(timestamp.to_string());
        std::fs::write(path.clone(), query).unwrap();
        info!("Wrote query to {}", path.to_string_lossy());

        let embedding = match embed(&EmbeddingSource {
            filepath: path.to_string_lossy().to_string(),
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

        let mut index_results = Vec::new();
        let result = self.index.query(&query, k, 200);

        index_results.extend(result.iter().map(|p| DeweyResponseItem {
            filepath: p.0.source_file.filepath.clone(),
            subset: match p.0.source_file.subset {
                Some(s) => s,
                None => (0, 0),
            },
        }));

        let response = DeweyResponse {
            results: index_results,
        };

        let response = match serde_json::to_string(&response) {
            Ok(serialized_response) => serialized_response,
            Err(e) => {
                error!("Failed to serialize response: {}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
            }
        };

        Ok(response)
    }

    // this returns an empty json object {} on success
    // or an object with just an `error` key on error
    pub fn reindex(&mut self, payload: RequestPayload) -> Result<String, std::io::Error> {
        let filepath = match payload {
            RequestPayload::Edit { filepath } => filepath,
            _ => {
                error!("malformed edit request: {:?}", payload);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "malformed edit request",
                ));
            }
        };

        let response = match crate::dbio::update_file_embeddings(&filepath, &mut self.index) {
            Ok(_) => "{}".to_string(),
            Err(e) => format!("{{error: {}}}", e),
        };

        Ok(response)
    }
}
