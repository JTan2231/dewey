use std::env;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread;
use unicode_segmentation::UnicodeSegmentation;

use crate::ledger::{get_indexing_rules, IndexRuleType};
use crate::logger::Logger;
use crate::{error, info, printl};

pub const EMBED_DIM: usize = 1536;

#[derive(Debug, Clone)]
struct RequestParams {
    host: String,
    path: String,
    port: u16,
    model: String,
    authorization_token: String,
}

#[derive(Debug, Clone)]
pub struct EmbeddingSource {
    pub filepath: String,
    pub subset: Option<(u64, u64)>,
}

impl EmbeddingSource {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let u64_size = std::mem::size_of::<u64>();

        let filepath_len = u64::from_be_bytes(bytes[0..u64_size].try_into().unwrap()) as usize;
        let filepath =
            String::from_utf8(bytes[u64_size..u64_size + filepath_len].to_vec()).unwrap();
        let start = u64::from_be_bytes(
            bytes[u64_size + filepath_len..u64_size + filepath_len + u64_size]
                .try_into()
                .unwrap(),
        );
        let end = u64::from_be_bytes(
            bytes[u64_size + filepath_len + u64_size..u64_size + filepath_len + 2 * u64_size]
                .try_into()
                .unwrap(),
        );

        let subset = match (start, end) {
            (0, 0) => None,
            _ => Some((start, end)),
        };

        Self { filepath, subset }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let filepath_len = self.filepath.len() as u64;

        bytes.extend_from_slice(&filepath_len.to_be_bytes());
        bytes.extend_from_slice(&self.filepath.as_bytes());
        bytes.extend_from_slice(&self.subset.unwrap_or((0, 0)).0.to_be_bytes());
        bytes.extend_from_slice(&self.subset.unwrap_or((0, 0)).1.to_be_bytes());

        bytes
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: u64,
    pub source_file: EmbeddingSource,
    pub data: [f32; 1536],
}

impl Embedding {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let u64_size = std::mem::size_of::<u64>();
        let f32_size = std::mem::size_of::<f32>();

        let mut offset = 0;
        let id = u64::from_be_bytes(bytes[0..u64_size].try_into().unwrap());
        offset += u64_size;

        let source_size = u64::from_be_bytes(bytes[offset..offset + u64_size].try_into().unwrap());
        offset += u64_size;

        let source = EmbeddingSource::from_bytes(&bytes[offset..offset + source_size as usize]);
        offset += source_size as usize;

        let mut data = [0.0; 1536];
        for (i, value) in bytes[offset..]
            .chunks(f32_size)
            .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
            .enumerate()
        {
            data[i] = value;
        }

        Self {
            id,
            source_file: source,
            data,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        let source_bytes = self.source_file.to_bytes();
        let data_bytes = self
            .data
            .iter()
            .map(|f| f.to_be_bytes())
            .collect::<Vec<_>>()
            .concat();

        bytes.extend_from_slice(&self.id.to_be_bytes());
        bytes.extend_from_slice(&(source_bytes.len() as u64).to_be_bytes());
        bytes.extend_from_slice(&source_bytes);
        bytes.extend_from_slice(&data_bytes);

        bytes
    }
}

fn read_source(source: &EmbeddingSource) -> Result<String, std::io::Error> {
    let mut file = match std::fs::File::open(&source.filepath) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open file: {:?}", e);
            return Err(e);
        }
    };

    let mut buffer = Vec::new();
    let contents = match file.read_to_end(&mut buffer) {
        Ok(_) => String::from_utf8_lossy(&buffer).to_string(),
        Err(e) => {
            error!("Failed to read from file {}: {:?}", source.filepath, e);
            return Err(e);
        }
    };

    // TODO: we can easily get away without loading the entire file into memory
    let contents = match source.subset {
        Some((start, end)) => contents[start as usize..end as usize].to_string(),
        _ => contents,
    };

    // CRLF -> LF
    let contents = contents.replace("\r\n", "\n");

    Ok(contents)
}

// TODO: a proper tokenizer
const TOKEN_LIMIT: usize = 8192;
fn separator_split(
    source: &EmbeddingSource,
    separator: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let contents = read_source(&source)?;
    let chars = contents.graphemes(true).collect::<Vec<&str>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() - separator.len() {
        let window = chars[i..i + separator.len()].join("");
        if window == *separator || chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();

            i += separator.len();
        } else {
            chunk.push_str(chars[i]);
            i += chars[i].len();
        }
    }

    if !chunk.is_empty() {
        chunks.push((
            chunk.clone(),
            (contents.len() - chunk.len(), contents.len()),
        ));
    }

    Ok(chunks)
}

// this only has a _separator argument so it can be used as a function pointer
// for either this or `separator_split`
fn naive_split(
    source: &EmbeddingSource,
    _separator: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let source_contents = read_source(&source)?;
    let chars = source_contents.graphemes(true).collect::<Vec<&str>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();
            i += 1;
        } else {
            chunk.push_str(chars[i]);
            i += chars[i].len();
        }
    }

    if !chunk.is_empty() {
        chunks.push((
            chunk.clone(),
            (source_contents.len() - chunk.len(), source_contents.len()),
        ));
    }

    Ok(chunks)
}

fn batch_sources(
    sources: &Vec<EmbeddingSource>,
) -> Result<Vec<Vec<(EmbeddingSource, String)>>, std::io::Error> {
    let indexing_rules = get_indexing_rules()?;
    info!("batching with rules: {:?}", indexing_rules);
    // API requests need batched up to keep from exceeding token limits
    let mut batches: Vec<Vec<(EmbeddingSource, String)>> = vec![Vec::new()];
    for source in sources {
        let extension = match source.filepath.split(".").last() {
            Some(extension) => extension,
            _ => {
                error!("Failed to get extension from file: {:?}", source.filepath);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Failed to get extension from file",
                ));
            }
        };

        info!("extension: {}", extension);
        let rules = indexing_rules.get(extension);
        let mut separator = "".to_string();
        let split_function: fn(
            &EmbeddingSource,
            &String,
        ) -> Result<Vec<(String, (usize, usize))>, std::io::Error> = match rules {
            Some(rules) => {
                let mut found = false;
                for rule in rules {
                    match rule.rule_type {
                        IndexRuleType::Split => {
                            separator = rule.value.clone();
                            found = true;
                        }
                        _ => (),
                    }
                }

                if !found {
                    info!("Using naive split for extension: {}", extension);
                    naive_split
                } else {
                    info!("Using separator split for extension: {}", extension);
                    separator_split
                }
            }
            _ => {
                info!("Using naive split for extension: {}", extension);
                naive_split
            }
        };

        let contents_split = split_function(&source, &separator)?;

        let mut split = batches.last_mut().unwrap();
        let mut split_len = 0;
        for (contents, window) in contents_split {
            if contents.len() + split_len >= TOKEN_LIMIT {
                batches.push(Vec::new());

                split = batches.last_mut().unwrap();
                split_len = 0;
            }

            if contents.len() > 0 {
                split_len += contents.len();
                let new_source = EmbeddingSource {
                    filepath: source.filepath.clone(),
                    subset: Some((window.0 as u64, window.1 as u64)),
                };
                split.push((new_source, contents));
            }
        }
    }

    batches.retain(|batch| !batch.is_empty());

    info!(
        "batched {} sources into {} batches",
        sources.len(),
        batches.len()
    );

    println!(
        "batched {} sources into {} batches",
        sources.len(),
        batches.len()
    );

    Ok(batches)
}

pub fn embed(sources: &Vec<EmbeddingSource>) -> Result<Vec<Embedding>, std::io::Error> {
    let params = RequestParams {
        host: "api.openai.com".to_string(),
        path: "/v1/embeddings".to_string(),
        port: 443,
        model: "text-embedding-3-small".to_string(),
        authorization_token: env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable not set"),
    };

    const NUM_THREADS: usize = 8;
    let mut thread_pool = Vec::new();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<(EmbeddingSource, String)>>();
    let rx = Arc::new(Mutex::new(rx));

    let embeddings = Arc::new(Mutex::new(Vec::new()));
    let count = Arc::new(Mutex::new(0));
    for i in 0..NUM_THREADS {
        let thread_rx = Arc::clone(&rx);
        let params = params.clone();
        let embeddings = Arc::clone(&embeddings);
        let count = Arc::clone(&count);
        let thread = thread::spawn(move || loop {
            let batch = thread_rx.lock().unwrap().recv();
            match batch {
                Ok(batch) => {
                    let success: Result<(), std::io::Error> = {
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

                        let connector = native_tls::TlsConnector::new()
                            .expect("Failed to create TLS connector");
                        let mut stream = connector
                            .connect(&params.host, stream)
                            .expect("Failed to establish TLS connection");

                        info!("Connected to OpenAI API");

                        let body = serde_json::json!({
                            "model": params.model,
                            "input": batch.iter().map(|pair| pair.1.clone()).collect::<Vec<String>>(),
                        });
                        let json = serde_json::json!(body);
                        let json_string = serde_json::to_string(&json)?;

                        let auth_string =
                            "Authorization: Bearer ".to_string() + &params.authorization_token;

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

                        info!("Sent request of size {} to OpenAI API", json_string.len());

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
                            error!("Failed to parse JSON: {}", body);
                            error!("Headers: {}", headers.join("\n"));
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "Failed to parse JSON",
                            ));
                        }

                        info!("Parsed JSON response");

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

                        {
                            let mut embeddings = embeddings.lock().unwrap();
                            for (i, datum) in data.iter().enumerate() {
                                let mut embedding = Embedding {
                                    id: 0,
                                    data: [0.0; 1536],
                                    source_file: batch[i].0.clone(),
                                };

                                for (i, value) in
                                    datum["embedding"].as_array().unwrap().iter().enumerate()
                                {
                                    embedding.data[i] = value.as_f64().unwrap() as f32;
                                }

                                embeddings.push(embedding);
                            }
                        }

                        Ok(())
                    };

                    thread::sleep(std::time::Duration::from_millis(250));

                    {
                        let mut count = count.lock().unwrap();
                        *count += 1;
                        if *count % 100 == 0 {
                            printl!(info, "{} embeddings made", *count);
                        }
                    }

                    match success {
                        Ok(_) => (),
                        Err(e) => {
                            error!("Failed to embed batch {}: {:?}", batch.len(), e);
                            return Err(e);
                        }
                    };
                }
                Err(_) => {
                    info!("Thread {} exiting", i);
                    return Ok(());
                }
            }
        });

        thread_pool.push(thread);
    }

    // API requests need batched up to keep from exceeding token limits
    let batches = batch_sources(&sources)?;
    printl!(info, "working through {} batches", batches.len());
    for batch in batches.iter() {
        tx.send(batch.clone()).unwrap();
    }

    drop(tx);

    for thread in thread_pool {
        thread.join().unwrap()?;
    }

    let embeddings = Arc::try_unwrap(embeddings).unwrap().into_inner().unwrap();
    Ok(embeddings)
}
