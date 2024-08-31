use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use dewey_lib::config;
use dewey_lib::hnsw::HNSW;
use dewey_lib::info;
use dewey_lib::logger::Logger;
use dewey_lib::message::Message;
use dewey_lib::openai::{embed, read_source, EmbeddingSource};
use dewey_lib::serialization::Serialize;

fn handle_client(mut stream: TcpStream, index: Arc<Mutex<HNSW>>) -> Result<(), std::io::Error> {
    let mut buffer = [0; 8192];
    stream.read(&mut buffer).unwrap();

    let (message, _) = Message::from_bytes(&buffer, 0).unwrap();
    info!("Received message: {}", message.body);

    let timestamp = chrono::Utc::now().timestamp_micros();
    let path = config::get_local_dir()
        .join("queries")
        .join(timestamp.to_string());
    std::fs::write(path.clone(), message.body)?;
    info!("Wrote query to {}", path.to_string_lossy());

    let embedding = embed(&vec![EmbeddingSource {
        filepath: path.to_string_lossy().to_string(),
        subset: None,
    }])?;

    #[allow(unused_assignments)]
    let mut index_result = String::new();
    {
        let index = index.lock().unwrap();
        let result = index.query(&embedding[0], 1, 200);

        index_result = read_source(&result[0].0.source_file)?;
    }

    let response = Message {
        message_type: "response".to_string(),
        body: index_result,
    };

    let response_bytes = response.to_bytes();
    stream.write(&response_bytes).unwrap();
    stream.flush().unwrap();

    Ok(())
}

pub fn main() -> std::io::Result<()> {
    config::setup();

    let listener = TcpListener::bind("127.0.0.1:5050").unwrap();
    info!("Server listening on port 5050");

    let index = Arc::new(Mutex::new(HNSW::new(false)?));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let index = Arc::clone(&index);
                thread::spawn(|| {
                    let _ = handle_client(stream, index);
                });
            }
            Err(e) => {
                info!("Error: {}", e);
            }
        }
    }

    Ok(())
}