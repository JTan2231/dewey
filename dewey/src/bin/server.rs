use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use dewey_lib::config;
use dewey_lib::hnsw::{Filter, Query, HNSW};
use dewey_lib::logger::Logger;
use dewey_lib::message::{DeweyRequest, DeweyResponse};
use dewey_lib::openai::{embed, EmbeddingSource};
use dewey_lib::parsing::read_source;
use dewey_lib::serialization::Serialize;
use dewey_lib::{error, info};

fn handle_client(mut stream: TcpStream, index: Arc<Mutex<HNSW>>) -> Result<(), std::io::Error> {
    let mut buffer = [0; 8192];
    stream.read(&mut buffer).unwrap();
    let buffer = String::from_utf8_lossy(&buffer).to_string();
    let buffer = buffer.trim_matches('\0');

    let message: DeweyRequest = match serde_json::from_str(&buffer) {
        Ok(msg) => msg,
        Err(e) => {
            error!("Failed to parse request: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
        }
    };

    let timestamp = chrono::Utc::now().timestamp_micros();
    let path = config::get_local_dir()
        .join("queries")
        .join(timestamp.to_string());
    std::fs::write(path.clone(), message.query).unwrap();
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

    let filters = message
        .filters
        .iter()
        .map(|f| Filter::from_string(&f.to_string()).unwrap())
        .collect::<Vec<Filter>>();

    let query = Query { embedding, filters };

    #[allow(unused_assignments)]
    let mut index_result = String::new();
    {
        let index = index.lock().unwrap();
        let result = index.query(&query, 1, 200);

        index_result = match read_source(&result[0].0.source_file) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read source file: {}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
            }
        }
    }

    let response = DeweyResponse { body: index_result };
    let response = match serde_json::to_string(&response) {
        Ok(serialized_response) => serialized_response,
        Err(e) => {
            error!("Failed to serialize response: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    };

    let mut bytes = Vec::new();
    bytes.extend((response.len() as u32).to_be_bytes());
    bytes.extend_from_slice(response.as_bytes());

    match stream.write(&response.to_bytes()) {
        Ok(bytes_written) => {
            stream.flush().unwrap();
            info!("wrote {} bytes to stream", bytes_written);
        }
        Err(e) => {
            error!("Failed to write response: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
        }
    }

    Ok(())
}

pub fn main() -> std::io::Result<()> {
    config::setup();

    let listener = TcpListener::bind("127.0.0.1:5051").unwrap();
    info!("Server listening on port 5051");

    let index = Arc::new(Mutex::new(HNSW::new(false)?));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let index = Arc::clone(&index);
                thread::spawn(|| match handle_client(stream, index) {
                    Ok(()) => {}
                    Err(e) => error!("Error handling client: {}", e),
                });
            }
            Err(e) => {
                info!("Error: {}", e);
            }
        }
    }

    Ok(())
}
