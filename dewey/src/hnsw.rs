use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

use serialize_macros::Serialize;

use crate::config::get_data_dir;
use crate::dbio::get_all_blocks;
use crate::logger::Logger;
use crate::openai::{Embedding, EMBED_DIM};
use crate::serialization::Serialize;
use crate::{info, printl};

pub fn dot(a: &Embedding, b: &Embedding) -> f32 {
    let mut sum = 0.;
    for i in 0..EMBED_DIM {
        sum += a.data[i] * b.data[i];
    }

    sum
}

pub fn normalize(embedding: &mut Embedding) {
    let mut sum = 0.;
    for i in 0..EMBED_DIM {
        sum += embedding.data[i] * embedding.data[i];
    }

    let sum = sum.sqrt();
    for i in 0..EMBED_DIM {
        embedding.data[i] /= sum;
    }
}

type Graph = HashMap<u64, Vec<(u64, f32)>>;

// basic in-memory nearest neighbor index
// TODO: should we handle huge datasets, beyond what memory can hold?
#[derive(Serialize)]
#[allow(unused_attributes)]
pub struct HNSW {
    embedding_sources: Vec<String>,
    // this should really go soon in favor of optimization
    // we can't really afford to have the whole db in memory, can we?
    #[ignore]
    nodes: Vec<Box<Embedding>>,
    pub layers: Vec<Graph>,
}

impl HNSW {
    pub fn new(reindex: bool) -> Result<Self, std::io::Error> {
        // TODO: boxing here instead of when they're loaded from file is gross
        //       and can probably be more efficient
        if !reindex {
            info!("loading index from disk");
            let data_dir = get_data_dir();
            let hnsw = Self::deserialize(data_dir.join("index").to_string_lossy().to_string())?;
            return Ok(hnsw);
        }

        info!("building index from block files");

        let block_embeddings = get_all_blocks()?;

        let mut embeddings = Vec::new();
        let mut sources = Vec::new();
        for be in block_embeddings {
            embeddings.push(be.embedding.clone());
            sources.push(be.source_file);
        }

        let id_map: HashMap<u64, usize> =
            HashMap::from_iter(embeddings.iter().enumerate().map(|(i, e)| (e.id, i)));

        let n = embeddings.len();
        let m = n.ilog2();
        let l = n.ilog2();
        let p = 1.0 / m as f32;

        printl!(
            info,
            "building HNSW with \n\tn: {}\n\tm: {}\n\tl: {}\n\tp: {}",
            n,
            m,
            l,
            p
        );

        let thresholds = (0..l)
            .map(|j| p * (1.0 - p).powi((j as i32 - l as i32 + 1).abs()))
            .collect::<Vec<_>>();

        // normalizing the probabilities since they don't usually add to 1
        let thresh_sum = thresholds.iter().sum::<f32>();
        let thresholds = thresholds
            .iter()
            .map(|&t| t / thresh_sum)
            .collect::<Vec<_>>();

        let mut orphans = HashSet::new();
        for i in 0..n {
            orphans.insert(i as u32);
        }

        let mut rng = thread_rng();
        let mut layers = vec![HashMap::new(); l as usize];
        for i in 0..n {
            if i % (n / 10) == 0 {
                printl!(
                    info,
                    "{} connected nodes, {} orphans, {} nodes attempted",
                    n - orphans.len(),
                    orphans.len(),
                    i + 1
                );
            }

            let prob = rng.gen::<f32>();
            for j in (0..l).rev() {
                let j = j as usize;
                // inserting the node into layer j
                // on insertion, form connections between the new node
                //               and the closest m neighbors in the layer
                //
                // each layer is a hashmap of ids to (node_id, distance) pairs
                // there's a gross mixing of using IDs and the actual embedding index here
                // this whole struct really needs a refactor
                if prob < thresholds[j] {
                    orphans.remove(&(i as u32));
                    for k in (j as u32)..l {
                        let k = k as usize;
                        let layer: &mut Graph = layers.get_mut(k).unwrap();

                        if layer.len() == 0 {
                            layer.insert(embeddings[i].id, Vec::new());
                        } else {
                            let distances: Vec<(u64, f32)> = layer
                                .keys()
                                .take(m as usize)
                                .map(|&node| {
                                    (
                                        node,
                                        1.0 - dot(
                                            embeddings[i].as_ref(),
                                            embeddings[id_map[&node]].as_ref(),
                                        ),
                                    )
                                })
                                .collect();

                            let mut updates = Vec::new();
                            for &(node, d) in distances.iter() {
                                let id = embeddings[i].id;
                                updates.push((node, id, d));
                                updates.push((id, node, d));
                            }

                            for (key, value, d) in updates {
                                let edges: &mut Vec<(u64, f32)> =
                                    layer.entry(key).or_insert_with(Vec::new).as_mut();
                                if !edges.contains(&(value, d)) {
                                    edges.push((value, d));
                                    edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                                }
                            }
                        }
                    }
                }
            }
        }

        printl!(info, "connecting {} orphans", orphans.len());
        let bottom_layer: &mut Graph = layers.get_mut(l as usize - 1).unwrap();
        for (i, orphan) in orphans.iter().enumerate() {
            if i % (orphans.len() / 10) == 0 {
                printl!(info, "{} orphans connected", i);
            }

            let orphan = *orphan as usize;
            let distances: Vec<(u64, f32)> = bottom_layer
                .keys()
                .take(m as usize)
                .map(|&node| {
                    (
                        node,
                        1.0 - dot(
                            embeddings[orphan].as_ref(),
                            embeddings[id_map[&node]].as_ref(),
                        ),
                    )
                })
                .collect();

            let mut updates = Vec::new();
            for &(node, d) in distances.iter() {
                let id = embeddings[orphan].id;
                updates.push((node, id, d));
                updates.push((id, node, d));
            }

            for (key, value, d) in updates {
                let edges: &mut Vec<(u64, f32)> =
                    bottom_layer.entry(key).or_insert_with(Vec::new).as_mut();
                if !edges.contains(&(value, d)) {
                    edges.push((value, d));
                    edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            }
        }

        printl!(info, "finished building index");

        Ok(Self {
            embedding_sources: sources,
            nodes: embeddings,
            layers,
        })
    }

    // should load files as needed
    // needs a rewrite
    pub fn query(&self, query: &Embedding, k: usize, ef: usize) -> Vec<(Box<Embedding>, f32)> {
        if ef < k {
            panic!("ef must be greater than k");
        }

        let mut visited = vec![false; self.nodes.len()];
        // frankly just a stupid way of using this instead of a min heap
        // but rust f32 doesn't have Eq so i don't know how to work with it
        let mut top_k: Vec<(u64, f32)> = Vec::new();

        let mut count = 0;
        let mut current = *self.layers[0].keys().next().unwrap();
        for layer in self.layers.iter() {
            let mut stack = Vec::new();
            stack.push(current);

            while !stack.is_empty() {
                current = stack.pop().unwrap();
                let mut neighbors = layer
                    .get(&current)
                    .unwrap()
                    .clone()
                    .into_iter()
                    .map(|(n, _)| (n, 1.0 - dot(query, self.nodes[n as usize].as_ref())))
                    .collect::<Vec<_>>();

                neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for (neighbor, distance) in neighbors {
                    let neighbor = neighbor as usize;
                    if !visited[neighbor] && count < ef {
                        top_k.push((neighbor as u64, distance));

                        stack.push(neighbor as u64);
                        visited[neighbor] = true;
                        count += 1;
                    }

                    if top_k.len() >= k {
                        top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        while top_k.len() >= k {
                            top_k.pop();
                        }
                    }

                    if count >= ef {
                        return top_k
                            .into_iter()
                            .map(|(node, distance)| (self.nodes[node as usize].clone(), distance))
                            .collect::<Vec<_>>();
                    }
                }
            }

            top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            current = top_k.first().unwrap().0;
        }

        top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        top_k
            .into_iter()
            .map(|(node, distance)| (self.nodes[node as usize].clone(), distance))
            .collect::<Vec<_>>()
    }

    pub fn serialize(&self, filepath: &String) -> Result<(), std::io::Error> {
        info!("serializing index to {}", filepath);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filepath)?;

        let bytes = self.to_bytes();
        file.write_all(&bytes)?;

        info!("finished serializing index");

        Ok(())
    }

    pub fn deserialize(filepath: String) -> Result<Self, std::io::Error> {
        printl!(info, "deserializing index from {}", filepath);

        let mut file = std::fs::File::open(filepath)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let (mut hnsw, _) = Self::from_bytes(&bytes, 0)?;

        let embedding_blocks = get_all_blocks()?;
        let embeddings = embedding_blocks
            .into_iter()
            .map(|be| be.embedding)
            .collect::<Vec<_>>();

        hnsw.nodes = embeddings;

        info!("finished deserializing index");

        Ok(hnsw)
    }

    pub fn print_graph(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {} has {} nodes", i, layer.len());
            for (node, neighbors) in layer.iter() {
                println!(
                    "  Node {}: {:?}",
                    node,
                    neighbors.iter().map(|(n, _)| n).collect::<Vec<_>>()
                );
            }
        }
    }
}
