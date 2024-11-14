use std::io::{Read, Write};

use crate::hnsw::{Filter, Query, HNSW};
use crate::logger::Logger;
use crate::message::{DeweyResponse, DeweyResponseItem, RequestPayload};
use crate::openai::{embed, EmbeddingSource};

mod cache;
pub mod config;
pub mod dbio;
pub mod hnsw;
pub mod ledger;
pub mod logger;
pub mod message;
mod openai;
mod parsing;
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

        info!("payload unpacked");

        let timestamp = chrono::Utc::now().timestamp_micros();
        let path = config::get_local_dir()
            .join("queries")
            .join(timestamp.to_string());
        match std::fs::write(path.clone(), query) {
            Ok(_) => {
                info!("Wrote query to {}", path.to_string_lossy());
            }
            Err(e) => {
                error!(
                    "error writing query to file {}: {}",
                    path.to_string_lossy(),
                    e
                );
                return Err(e);
            }
        };

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

        info!("embedding created");

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

pub struct DeweyClient {
    pub address: String,
    pub port: u32,
}

impl DeweyClient {
    pub fn new(address: String, port: u32) -> Self {
        Self { address, port }
    }

    fn send(
        &self,
        message: message::DeweyRequest,
    ) -> Result<message::DeweyResponse, std::io::Error> {
        let destination = format!("{}:{}", self.address, self.port);
        let mut stream = std::net::TcpStream::connect(destination.clone())?;

        let message = serde_json::to_string(&message)?;
        let mut bytes = Vec::new();
        bytes.extend((message.len() as u32).to_be_bytes());
        bytes.extend_from_slice(message.as_bytes());

        match stream.write(&bytes) {
            Ok(_) => {
                stream.flush().unwrap();
            }
            Err(e) => {
                error!("Failed to write response: {}", e);
                return Err(e);
            }
        };

        let mut length_bytes = [0u8; 4];
        stream.read_exact(&mut length_bytes)?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        let mut buffer = vec![0u8; length];
        stream.read_exact(&mut buffer)?;
        let buffer = String::from_utf8_lossy(&buffer);

        match serde_json::from_str(&buffer) {
            Ok(resp) => Ok(resp),
            Err(e) => {
                error!("Failed to parse response: {}", e);
                error!("buffer: {:?}", buffer);
                return Err(e.into());
            }
        }
    }

    pub fn query(
        &self,
        request: String,
        k: usize,
        filters: Vec<String>,
    ) -> Result<message::DeweyResponse, std::io::Error> {
        let message = message::DeweyRequest {
            message_type: "query".to_string(),
            payload: message::RequestPayload::Query {
                query: request,
                k,
                filters,
            },
        };

        self.send(message)
    }

    pub fn reindex(&self, filepath: String) -> Result<message::DeweyResponse, std::io::Error> {
        let message = message::DeweyRequest {
            message_type: "edit".to_string(),
            payload: message::RequestPayload::Edit { filepath },
        };

        self.send(message)
    }
}
