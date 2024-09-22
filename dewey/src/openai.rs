use std::env;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread;

use serialize_macros::Serialize;

use crate::ledger::{get_indexing_rules, IndexRuleType};
use crate::logger::Logger;
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

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingSource {
    pub filepath: String,
    pub subset: Option<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Embedding {
    pub id: u64,
    pub source_file: EmbeddingSource,
    pub data: [f32; EMBED_DIM],
}

pub fn read_source(source: &EmbeddingSource) -> Result<String, std::io::Error> {
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
    let chars = contents.chars().collect::<Vec<char>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() - separator.len() {
        let window = String::from_iter(&chars[i..i + separator.len()]);
        if window == *separator || chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();

            i += separator.len();
        } else {
            chunk.push_str(chars[i].to_string().as_str());
            i += 1;
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
    let chars = source_contents.chars().collect::<Vec<_>>();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chunk.len() >= TOKEN_LIMIT {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();
            i += 1;
        } else {
            chunk.push_str(chars[i].to_string().as_str());
            i += 1;
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

fn max_length_split(
    source: &EmbeddingSource,
    max_length: &String,
) -> Result<Vec<(String, (usize, usize))>, std::io::Error> {
    let source_contents = read_source(&source)?;
    let chars = source_contents.chars().collect::<Vec<_>>();
    let max_length = max_length.parse::<usize>().unwrap();

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chunk.len() >= TOKEN_LIMIT || chunk.len() >= max_length {
            chunks.push((chunk.clone(), (i - chunk.len(), i)));
            chunk.clear();
            i += 1;
        } else {
            chunk.push_str(chars[i].to_string().as_str());
            i += 1;
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

// NOTE: does _not_ support anything but ascii
fn batch_sources(
    sources: &Vec<EmbeddingSource>,
) -> Result<Vec<Vec<(EmbeddingSource, String)>>, std::io::Error> {
    let indexing_rules = get_indexing_rules()?;
    let base = Vec::new();
    let global_rules = indexing_rules.get("*").unwrap_or(&base);
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

        let mut rules = global_rules.clone();
        if let Some(extension_rules) = indexing_rules.get(extension) {
            rules.extend(extension_rules.clone());
        }

        let mut rule_arg = "".to_string();
        let split_function: fn(
            &EmbeddingSource,
            &String,
        ) -> Result<Vec<(String, (usize, usize))>, std::io::Error> = {
            let mut rule_type = "".to_string();
            for rule in rules.iter() {
                match rule.rule_type {
                    IndexRuleType::Split => {
                        rule_arg = rule.value.clone();
                        rule_type = "separator".to_string();
                    }
                    IndexRuleType::MaxLength => {
                        rule_arg = rule.value.clone();
                        rule_type = "max_length".to_string();
                    }
                    _ => (),
                }
            }

            match rule_type.as_str() {
                "separator" => separator_split,
                "max_length" => max_length_split,
                _ => naive_split,
            }
        };

        let mut contents_split = split_function(&source, &rule_arg)?;

        // there's probably a better way to apply these filters
        // in conjunction with the splitters
        for rule in rules {
            match rule.rule_type {
                IndexRuleType::MinLength => {
                    let min_length = rule.value.parse::<usize>().unwrap();
                    contents_split.retain(|(_, range)| range.1 - range.0 >= min_length);
                }
                IndexRuleType::Alphanumeric => {
                    contents_split.retain(|(contents, _)| {
                        contents
                            .chars()
                            .any(|c| c.is_alphanumeric() || c.is_whitespace())
                    });
                }
                _ => (),
            }
        }

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

pub fn embed_bulk(sources: &Vec<EmbeddingSource>) -> Result<Vec<Embedding>, std::io::Error> {
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
                            info!("{} embeddings made", *count);
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

    info!("working through {} batches", batches.len());
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

pub fn embed(source: &EmbeddingSource) -> Result<Embedding, std::io::Error> {
    let params = RequestParams {
        host: "api.openai.com".to_string(),
        path: "/v1/embeddings".to_string(),
        port: 443,
        model: "text-embedding-3-small".to_string(),
        authorization_token: env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable not set"),
    };

    let query = read_source(source)?;
    if query.len() == 0 || query.len() > TOKEN_LIMIT {
        error!("Invalid query size: {}", query.len());
        error!("Query must be between 1 and {} characters", TOKEN_LIMIT);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Failed to read source",
        ));
    }

    // TODO: a lot of this is just copy+paste code
    //       should be abstracted i think
    let success: Result<Vec<Embedding>, std::io::Error> = {
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
            "input": query,
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
                error!("Failed to parse data from JSON: {:?}", response_json);
                error!("Request: {}", request);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to parse data from JSON",
                ));
            }
        };

        let mut embeddings = Vec::new();
        for datum in data.iter() {
            let mut embedding = Embedding {
                id: 0,
                data: [0.0; 1536],
                source_file: source.clone(),
            };

            for (i, value) in datum["embedding"].as_array().unwrap().iter().enumerate() {
                embedding.data[i] = value.as_f64().unwrap() as f32;
            }

            embeddings.push(embedding);
        }

        Ok(embeddings)
    };

    match success {
        Ok(embeddings) => Ok(embeddings[0].clone()),
        Err(e) => {
            error!("Failed to embed query \"{}\": {:?}", query, e);
            return Err(e);
        }
    }
}
