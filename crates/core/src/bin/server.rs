use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use dewey_lib::config;
use dewey_lib::hnsw::{Filter, Query, HNSW};
use dewey_lib::logger::Logger;
use dewey_lib::message::{DeweyRequest, DeweyResponse, DeweyResponseItem};
use dewey_lib::openai::{embed, EmbeddingSource};
use dewey_lib::serialization::Serialize;
use dewey_lib::{error, info};

struct Flags {
    address: String,
    port: usize,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        address: String::from("127.0.0.1"),
        port: 5050,
    };

    if args.len() < 1 {
        std::process::exit(1);
    }

    for (i, arg) in args.iter().skip(1).enumerate() {
        if arg.starts_with("-") && !arg.starts_with("--") {
            for c in arg.chars().skip(1) {
                match c {
                    'a' => {
                        flags.address = args[i + 2].clone();
                    }
                    'p' => {
                        flags.port = args[i + 2].parse().unwrap();
                    }
                    _ => panic!("error: unknown flag: {}", c),
                }
            }
        }
    }

    flags
}

fn handle_client(mut stream: TcpStream, index: Arc<Mutex<HNSW>>) -> Result<(), std::io::Error> {
    let mut size_buffer = [0u8; 4];
    stream.read_exact(&mut size_buffer).unwrap();
    let message_size = u32::from_be_bytes(size_buffer) as usize;

    let mut buffer = vec![0u8; message_size];
    stream.read_exact(&mut buffer).unwrap();

    let request = String::from_utf8_lossy(&buffer);

    let message: DeweyRequest = match serde_json::from_str(&request) {
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
    let mut index_results = Vec::new();
    {
        let index = index.lock().unwrap();
        let result = index.query(&query, message.k, 200);

        index_results.extend(result.iter().map(|p| DeweyResponseItem {
            filepath: p.0.source_file.filepath.clone(),
            subset: match p.0.source_file.subset {
                Some(s) => s,
                None => (0, 0),
            },
        }));
    }

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
    let flags = parse_flags();

    let listener = TcpListener::bind(format!("{}:{}", flags.address, flags.port)).unwrap();
    info!("Server listening on {}:{}", flags.address, flags.port);
    println!("Server listening on {}:{}", flags.address, flags.port);

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
