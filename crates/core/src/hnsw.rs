use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

use serialize_macros::Serialize;

use crate::cache::EmbeddingCache;
use crate::config::get_data_dir;
use crate::dbio::{get_directory, BLOCK_SIZE};
use crate::info;
use crate::logger::Logger;
use crate::openai::{Embedding, EMBED_DIM};
use crate::serialization::Serialize;

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

const CACHE_SIZE: u32 = 20 * BLOCK_SIZE as u32;

pub enum FilterComparator {
    Equal,
    NotEqual,
}

pub struct Filter {
    pub comparator: FilterComparator,
    pub value: String,
}

impl Filter {
    pub fn from_string(input: &String) -> Result<Self, std::io::Error> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid filter format",
            ));
        }

        let comparator = match parts[0] {
            "eq" => FilterComparator::Equal,
            "ne" => FilterComparator::NotEqual,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Invalid comparator",
                ))
            }
        };

        Ok(Filter {
            comparator,
            value: parts[1].to_string(),
        })
    }

    pub fn compare(self: &Self, query: &str) -> bool {
        match self.comparator {
            FilterComparator::Equal => query == self.value,
            FilterComparator::NotEqual => query != self.value,
        }
    }
}

pub struct Query {
    pub embedding: Embedding,
    pub filters: Vec<Filter>,
}

// basic in-memory nearest neighbor index
// TODO: should we handle huge datasets, beyond what memory can hold?
#[derive(Serialize)]
#[allow(unused_attributes)]
pub struct HNSW {
    pub size: u32,
    pub layers: Vec<Graph>,
}

impl HNSW {
    pub fn new(reindex: bool) -> Result<Self, std::io::Error> {
        if !reindex {
            info!("loading index from disk");
            let data_dir = get_data_dir();
            let hnsw = Self::deserialize(data_dir.join("index").to_string_lossy().to_string())?;
            return Ok(hnsw);
        }

        info!("building index from block files");

        let directory = get_directory()?;

        let n = directory.len();
        let m = n.ilog2();
        let l = n.ilog2();
        let p = 1.0 / m as f32;

        info!(
            "building HNSW with \n\tn: {}\n\tm: {}\n\tl: {}\n\tp: {}",
            n, m, l, p
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

        // TODO: config param?
        let mut cache = EmbeddingCache::new(CACHE_SIZE);

        let mut rng = thread_rng();
        let mut layers = vec![HashMap::new(); l as usize];
        // for each embedding e[i]
        for i in 0..n {
            if i % (n / 10) == 0 {
                info!(
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
                    let e_i = cache.get(i as u32)?;

                    for k in (j as u32)..l {
                        let k = k as usize;
                        let layer: &mut Graph = layers.get_mut(k).unwrap();

                        if layer.len() == 0 {
                            layer.insert(e_i.id, Vec::new());
                        } else {
                            let distances: Vec<(u64, f32)> = layer
                                .keys()
                                .take(m as usize)
                                .map(|&neighbor| {
                                    let e_neighbor = cache.get(neighbor as u32).unwrap();
                                    (neighbor, 1.0 - dot(&e_i, &e_neighbor))
                                })
                                .collect();

                            let mut updates = Vec::new();
                            for &(node, d) in distances.iter() {
                                updates.push((node, e_i.id, d));
                                updates.push((e_i.id, node, d));
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

        info!("connecting {} orphans", orphans.len());
        let bottom_layer: &mut Graph = layers.get_mut(l as usize - 1).unwrap();

        // sorting here makes better use of the cache + how embeddings are loaded
        let mut orphans = orphans.iter().collect::<Vec<_>>();
        orphans.sort_by(|a, b| a.cmp(b));
        for (i, orphan) in orphans.iter().enumerate() {
            if i % (orphans.len() / 10) == 0 {
                info!("{} orphans connected", i);
            }

            let orphan = **orphan as usize;
            let e_orphan = cache.get(orphan as u32)?;
            let distances: Vec<(u64, f32)> = bottom_layer
                .keys()
                .take(m as usize)
                .map(|&neighbor| {
                    let e_neighbor = cache.get(neighbor as u32).unwrap();
                    (neighbor, 1.0 - dot(&e_orphan, &e_neighbor))
                })
                .collect();

            let mut updates = Vec::new();
            for &(node, d) in distances.iter() {
                updates.push((node, e_orphan.id, d));
                updates.push((e_orphan.id, node, d));
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

        info!("finished building index");

        Ok(Self {
            size: n as u32,
            layers,
        })
    }

    // please god optimize this
    // is this better than bfs?
    //
    // dfs search through the hnsw
    pub fn query(&self, query: &Query, k: usize, ef: usize) -> Vec<(Box<Embedding>, f32)> {
        if ef < k {
            panic!("ef must be greater than k");
        }

        // there's gotta be a better way to blacklist
        let mut visited = vec![false; self.size as usize];
        let mut blacklist = vec![false; self.size as usize];

        // frankly just a stupid way of using this instead of a min heap
        // but rust f32 doesn't have Eq so i don't know how to work with it
        let mut top_k: Vec<(u64, f32)> = Vec::new();

        let mut cache = EmbeddingCache::new(CACHE_SIZE);

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
                    .filter_map(|(n, _)| {
                        if blacklist[n as usize] {
                            return None;
                        }

                        let e_n = cache.get(n as u32).unwrap();
                        let mut filter_pass = true;
                        for filter in query.filters.iter() {
                            for meta in e_n.source_file.meta.iter() {
                                filter_pass &= filter.compare(meta);
                            }
                        }

                        if !visited[n as usize] && filter_pass {
                            Some((n, 1.0 - dot(&query.embedding, &e_n)))
                        } else {
                            blacklist[n as usize] = true;
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for (neighbor, distance) in neighbors {
                    let neighbor = neighbor as usize;
                    if !visited[neighbor] && !blacklist[neighbor] && count < ef {
                        top_k.push((neighbor as u64, distance));

                        stack.push(neighbor as u64);
                        visited[neighbor] = true;
                        count += 1;
                    }

                    if top_k.len() > k {
                        top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        while top_k.len() > k {
                            top_k.pop();
                        }
                    }

                    if count >= ef {
                        return top_k
                            .into_iter()
                            .map(|(node, distance)| (cache.get(node as u32).unwrap(), distance))
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
            .map(|(node, distance)| (cache.get(node as u32).unwrap(), distance))
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
        info!("deserializing index from {}", filepath);

        let mut file = std::fs::File::open(filepath)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let (hnsw, count) = Self::from_bytes(&bytes, 0)?;

        if count <= 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid index file",
            ));
        }

        info!("finished deserializing index");

        Ok(hnsw)
    }

    pub fn get_last_layer(&self) -> &Graph {
        self.layers.last().unwrap()
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
