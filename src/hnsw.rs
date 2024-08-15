use crate::dbio::EmbeddingBlock;
use crate::info;
use crate::logger::Logger;
use crate::openai::{Embedding, EMBED_DIM};

use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};

fn dot(a: &Embedding, b: &Embedding) -> f32 {
    let mut sum = 0.;
    for i in 0..EMBED_DIM {
        sum += a.data[i] * b.data[i];
    }

    sum
}

type Graph = HashMap<u32, Vec<(u32, f32)>>;

// basic in-memory nearest neighbor index
// TODO: should we handle huge datasets, beyond what memory can hold?
pub struct HNSW {
    nodes: Vec<Box<Embedding>>,
    layers: Vec<Graph>,
}

impl HNSW {
    pub fn new(block_number: u64) -> Self {
        // TODO: boxing here instead of when they're loaded from file is gross
        //       and can probably be more efficient
        let block = EmbeddingBlock::from_file(&block_number.to_string()).unwrap();
        let embeddings = block
            .embeddings
            .into_iter()
            .map(Box::new)
            .collect::<Vec<_>>();

        let n = embeddings.len();
        let m = n.ilog2();
        let l = n.ilog2();
        let p = 1.0 / m as f32;

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
            let prob = rng.gen::<f32>();
            info!("node {} prob {}", i, prob);
            for j in (0..l).rev() {
                let j = j as usize;
                info!("layer {} probablity {}", j, thresholds[j]);
                // inserting the node into layer j
                // on insertion, form connections between the new node
                //               and the closest m neighbors in the layer
                if prob < thresholds[j] {
                    orphans.remove(&(i as u32));
                    for k in (j as u32)..l {
                        let k = k as usize;
                        let layer: &mut Graph = layers.get_mut(k).unwrap();

                        if layer.len() == 0 {
                            layer.insert(i as u32, Vec::new());
                        } else {
                            let i = i as u32;

                            let mut distances: Vec<(u32, f32)> = layer
                                .keys()
                                .map(|&node| {
                                    (
                                        node,
                                        dot(
                                            embeddings[i as usize].as_ref(),
                                            embeddings[node as usize].as_ref(),
                                        ),
                                    )
                                })
                                .collect();

                            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                            let mut updates = Vec::new();
                            for &(node, d) in distances.iter().take(m as usize) {
                                updates.push((node, i, d));
                                updates.push((i, node, d));
                            }

                            for (key, value, d) in updates {
                                let edges: &mut Vec<(u32, f32)> =
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

        let bottom_layer: &mut Graph = layers.get_mut(l as usize - 1).unwrap();
        for orphan in orphans {
            let mut distances: Vec<(u32, f32)> = bottom_layer
                .keys()
                .map(|&node| {
                    (
                        node,
                        dot(
                            embeddings[orphan as usize].as_ref(),
                            embeddings[node as usize].as_ref(),
                        ),
                    )
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let mut updates = Vec::new();
            for &(node, d) in distances.iter().take(m as usize) {
                updates.push((node, orphan, d));
                updates.push((orphan, node, d));
            }

            for (to, from, d) in updates {
                let edges: &mut Vec<(u32, f32)> =
                    bottom_layer.entry(to).or_insert_with(Vec::new).as_mut();
                if !edges.contains(&(from, d)) {
                    edges.push((from, d));
                    edges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            }
        }

        Self {
            nodes: embeddings,
            layers,
        }
    }

    pub fn query(&self, query: &Embedding, k: usize, ef: usize) -> Vec<Box<Embedding>> {
        let mut top_k = Vec::new();
        let mut current = self.layers[0].keys().next().unwrap();

        top_k
    }

    pub fn print_graph(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {} has {} nodes", i, layer.len());
            for (node, neighbors) in layer.iter() {
                println!(
                    "  Node {}: {:?}",
                    node,
                    neighbors.iter().map(|(n, d)| n).collect::<Vec<_>>()
                );
            }
        }
    }
}
