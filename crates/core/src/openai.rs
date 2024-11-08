use std::env;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread;

use rand::Rng;

use serialize_macros::Serialize;

use crate::logger::Logger;
use crate::parsing::{batch_sources, read_source, TOKEN_LIMIT};
use crate::serialization::Serialize;
use crate::{error, info};

pub const EMBED_DIM: usize = 1536;

#[derive(Debug, Clone)]
struct RequestParams {
    host: String,
    path: String,
    port: u16,
    model: String,
    authorization_token: String,
}

impl RequestParams {
    fn new() -> Self {
        Self {
            host: "api.openai.com".to_string(),
            path: "/v1/embeddings".to_string(),
            port: 443,
            model: "text-embedding-3-small".to_string(),
            authorization_token: env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY environment variable not set"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingSource {
    pub filepath: String,
    pub meta: std::collections::HashSet<String>,
    pub subset: Option<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Embedding {
    pub id: u64,
    pub source_file: EmbeddingSource,
    pub data: [f32; EMBED_DIM],
}

trait EmbeddingApiClient {
    fn embedding_api_call(
        params: &RequestParams,
        batch: &Vec<(EmbeddingSource, String)>,
    ) -> Result<Vec<Embedding>, std::io::Error>;
}

struct ApiClient;
impl EmbeddingApiClient for ApiClient {
    fn embedding_api_call(
        params: &RequestParams,
        batch: &Vec<(EmbeddingSource, String)>,
    ) -> Result<Vec<Embedding>, std::io::Error> {
        let duration = std::time::Duration::from_secs(30);
        let address = (params.host.clone(), params.port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| {
                error!(
                    "Failed to resolve address {:?}",
                    (params.host.clone(), params.port)
                );
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Failed to resolve address",
                )
            })?;

        let stream = match TcpStream::connect_timeout(&address, duration) {
            Ok(stream) => stream,
            Err(e) => {
                error!("Failed to connect to OpenAI API: {:?}", e);
                return Err(e);
            }
        };

        match stream.set_read_timeout(Some(duration)) {
            Ok(_) => (),
            Err(e) => {
                error!("Failed to set read timeout: {:?}", e);
                return Err(e);
            }
        }

        match stream.set_write_timeout(Some(duration)) {
            Ok(_) => (),
            Err(e) => {
                error!("Failed to set write timeout: {:?}", e);
                return Err(e);
            }
        }

        let connector = native_tls::TlsConnector::new().expect("Failed to create TLS connector");
        let mut stream = connector
            .connect(&params.host, stream)
            .expect("Failed to establish TLS connection");

        let body = serde_json::json!({
            "model": params.model,
            "input": batch.iter().map(|pair| pair.1.clone()).collect::<Vec<String>>(),
        });
        let json = serde_json::json!(body);
        let json_string = serde_json::to_string(&json)?;

        let auth_string = "Authorization: Bearer ".to_string() + &params.authorization_token;

        let request = format!(
            "POST {} HTTP/1.1\r\n\
        Host: {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Accept: */*\r\n\
        {}\r\n\r\n\
        {}",
            params.path,
            params.host,
            json_string.len(),
            auth_string,
            json_string.trim()
        );

        match stream.write_all(request.as_bytes()) {
            Ok(_) => (),
            Err(e) => {
                error!("Failed to write to OpenAI stream: {:?}", e);
                return Err(e);
            }
        }

        match stream.flush() {
            Ok(_) => (),
            Err(e) => {
                error!("Failed to flush OpenAI stream: {:?}", e);
                return Err(e);
            }
        }

        let mut reader = std::io::BufReader::new(&mut stream);

        let mut buffer = String::new();
        // read 2 characters at a time to check for CRLF
        while !buffer.ends_with("\r\n\r\n") {
            let mut chunk = [0; 1];
            match reader.read(&mut chunk) {
                Ok(0) => {
                    error!("Failed to read from OpenAI stream: EOF");
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Failed to read from OpenAI stream",
                    ));
                }
                Ok(_) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                }
                Err(e) => {
                    error!("Failed to read from OpenAI stream: {:?}", e);
                    return Err(e);
                }
            }
        }

        let headers = buffer.split("\r\n").collect::<Vec<&str>>();
        let content_length = headers
            .iter()
            .find(|header| header.starts_with("Content-Length"))
            .ok_or_else(|| {
                error!("Failed to find Content-Length header: {:?}", headers);
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to find Content-Length header",
                )
            })?;

        let content_length = content_length.split(": ").collect::<Vec<&str>>()[1]
            .parse::<usize>()
            .unwrap();

        let mut body = vec![0; content_length];
        reader.read_exact(&mut body)?;

        let body = String::from_utf8_lossy(&body).to_string();
        let response_json = serde_json::from_str(&body);

        if response_json.is_err() {
            error!("request: {}", request);
            error!("Failed to parse JSON: {}", body);
            error!("Headers: {}", headers.join("\n"));
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to parse JSON",
            ));
        }

        let response_json: serde_json::Value = response_json.unwrap();
        let data = match response_json["data"].as_array() {
            Some(data) => data,
            _ => {
                error!("batch: {:?}", batch);
                error!("Failed to parse data from JSON: {:?}", response_json);
                error!("Request: {}", request);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to parse data from JSON",
                ));
            }
        };

        let mut embeddings = Vec::new();
        for (i, datum) in data.iter().enumerate() {
            let mut embedding = Embedding {
                id: 0,
                data: [0.0; 1536],
                source_file: batch[i].0.clone(),
            };

            for (i, value) in datum["embedding"].as_array().unwrap().iter().enumerate() {
                embedding.data[i] = value.as_f64().unwrap() as f32;
            }

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

struct TestApiCall;
impl EmbeddingApiClient for TestApiCall {
    fn embedding_api_call(
        _params: &RequestParams,
        batch: &Vec<(EmbeddingSource, String)>,
    ) -> Result<Vec<Embedding>, std::io::Error> {
        let mut embeddings = Vec::new();
        let mut rng = rand::thread_rng();
        for (i, b) in batch.iter().enumerate() {
            let embedding = Embedding {
                id: i as u64,
                data: [0.0; 1536].map(|_| rng.gen()),
                source_file: b.0.clone(),
            };

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

// multithreaded wrapper over the actual bulk API call
pub fn embed_bulk(sources: &Vec<EmbeddingSource>) -> Result<Vec<Embedding>, std::io::Error> {
    let params = RequestParams::new();

    // there's probably a better programmatic way of determining this
    const NUM_THREADS: usize = 8;
    let mut thread_pool = Vec::new();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<(EmbeddingSource, String)>>();
    let rx = Arc::new(Mutex::new(rx));

    // API requests need batched up to keep from exceeding token limits
    let batches = batch_sources(&sources)?;

    let embeddings = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::new(Mutex::new(0));
    for i in 0..std::cmp::min(NUM_THREADS, batches.len()) {
        let thread_rx = Arc::clone(&rx);
        let params = params.clone();
        let embeddings = Arc::clone(&embeddings);
        let count = Arc::clone(&count);
        let thread = thread::spawn(move || loop {
            let batch = thread_rx.lock().unwrap().recv();
            match batch {
                Ok(batch) => {
                    match ApiClient::embedding_api_call(&params, &batch) {
                        Ok(new_embeddings) => {
                            let mut embeddings = embeddings.lock().unwrap();
                            embeddings.extend(new_embeddings);

                            let mut count = count.lock().unwrap();
                            *count += 1;
                            if *count % 100 == 0 {
                                info!("{} embeddings made", *count);
                            }
                        }
                        Err(e) => {
                            error!("Failed to embed batch {}: {:?}", batch.len(), e);
                            continue;
                        }
                    };
                }
                Err(_) => {
                    info!("Thread {} exiting", i);
                }
            }
        });

        thread_pool.push(thread);
    }

    info!("working through {} batches", batches.len());

    // TODO: figure out a process for dealing with failed batches
    let mut retries = 0;
    for batch in batches.iter() {
        while let Err(e) = tx.send(batch.clone()) {
            error!("failed to send batch: {}", e);

            // arbitrary limit
            if retries >= 5 {
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
            retries += 1;
        }
    }

    drop(tx);

    for thread in thread_pool {
        thread.join().unwrap();
    }

    let embeddings = Arc::try_unwrap(embeddings).unwrap().into_inner().unwrap();
    Ok(embeddings)
}

pub fn embed(source: &EmbeddingSource) -> Result<Embedding, std::io::Error> {
    let query = read_source(source)?;
    if query.len() == 0 || query.len() > TOKEN_LIMIT {
        error!("Invalid query size: {}", query.len());
        error!("Query must be between 1 and {} characters", TOKEN_LIMIT);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Failed to read source",
        ));
    }

    match ApiClient::embedding_api_call(
        &RequestParams::new(),
        &vec![(source.clone(), query.clone())],
    ) {
        Ok(embeddings) => Ok(embeddings[0].clone()),
        Err(e) => {
            error!("Failed to embed query \"{}\": {:?}", query, e);
            return Err(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_call_test() {
        // TODO: implementation pending batch testing
    }
}
