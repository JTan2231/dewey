use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::config;
use crate::dbio::{EmbeddingBlock, BLOCK_SIZE};
use crate::logger::Logger;
use crate::openai::Embedding;
use crate::{error, info};

// most of this is ripped from https://rust-unofficial.github.io/too-many-lists/sixth-final.html
// this really could use some cleaning up
pub struct LinkedList<T> {
    front: Link<T>,
    back: Link<T>,
    len: usize,
    _boo: PhantomData<T>,
}

type Link<T> = Option<NonNull<Node<T>>>;

pub struct Node<T> {
    elem: T,
    front: Link<T>,
    back: Link<T>,
}

pub struct Iter<'a, T> {
    front: Link<T>,
    back: Link<T>,
    len: usize,
    _boo: PhantomData<&'a T>,
}

pub struct IterMut<'a, T> {
    front: Link<T>,
    back: Link<T>,
    len: usize,
    _boo: PhantomData<&'a mut T>,
}

pub struct IntoIter<T> {
    list: LinkedList<T>,
}

impl<T: Clone> Node<T> {
    // the fact that this doesn't alter the list len is really stupid
    pub fn detach(&mut self) -> T {
        if let Some(front) = self.front {
            unsafe {
                (*front.as_ptr()).back = self.back;
            }
        }

        if let Some(back) = self.back {
            unsafe {
                (*back.as_ptr()).front = self.front;
            }
        }

        self.elem.clone()
    }
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList {
            front: None,
            back: None,
            len: 0,
            _boo: PhantomData,
        }
    }

    pub fn push_front(&mut self, elem: T) -> NonNull<Node<T>> {
        unsafe {
            let new = NonNull::new_unchecked(Box::into_raw(Box::new(Node {
                front: None,
                back: None,
                elem,
            })));

            if let Some(old) = self.front {
                (*old.as_ptr()).front = Some(new);
                (*new.as_ptr()).back = Some(old);
            } else {
                self.back = Some(new);
            }

            self.front = Some(new);
            self.len += 1;
        }

        self.front.unwrap()
    }

    pub fn pop_back(&mut self) -> Option<T> {
        unsafe {
            self.back.map(|node| {
                let boxed_node = Box::from_raw(node.as_ptr());
                let result = boxed_node.elem;

                self.back = boxed_node.front;
                if let Some(new) = self.back {
                    (*new.as_ptr()).back = None;
                } else {
                    self.front = None;
                }

                if self.len > 0 {
                    self.len -= 1;
                }
                result
            })
        }
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            front: self.front,
            back: self.back,
            len: self.len,
            _boo: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            front: self.front,
            back: self.back,
            len: self.len,
            _boo: PhantomData,
        }
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        while self.pop_back().is_some() {}
    }
}

// zero clue how these iterator impls work
impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        // While self.front == self.back is a tempting condition to check here,
        // it won't do the right for yielding the last element! That sort of
        // thing only works for arrays because of "one-past-the-end" pointers.
        if self.len > 0 {
            // We could unwrap front, but this is safer and easier
            self.front.map(|node| unsafe {
                self.len -= 1;
                self.front = (*node.as_ptr()).back;
                &mut (*node.as_ptr()).elem
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            self.front.map(|node| unsafe {
                self.len -= 1;
                self.front = (*node.as_ptr()).back;
                &(*node.as_ptr()).elem
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T: Debug> Debug for LinkedList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

// lru: a list of embedding ids
// node_map: a map of embedding ids to their corresponding nodes in the lru
// embeddings: a map of embedding ids to their corresponding embeddings
//
// this relies on $DATA_DIR/directory to find indexed embeddings
//
// TODO: some sort of serialization for the cache
//       but is it even worth it? how bad are cold starts?
pub struct EmbeddingCache {
    lru: LinkedList<u32>,
    node_map: HashMap<u32, NonNull<Node<u32>>>,
    embeddings: HashMap<u32, Embedding>,
    directory: HashMap<u32, u64>,
    // ideally this is some multiple of the number of embeddings in a block
    // this _must_ be greater or equal to the number of embeddings in a block
    max_size: u32,
}

impl EmbeddingCache {
    pub fn new(max_size: u32) -> Self {
        info!("initializing embedding cache with max size {}", max_size);

        if max_size < BLOCK_SIZE as u32 {
            error!(
                "max_size {} must be greater than or equal to the number of embeddings in a block",
                max_size
            );
            panic!("max_size must be greater than or equal to the number of embeddings in a block");
        }

        let directory_path = format!("{}/directory", config::get_data_dir().to_str().unwrap());
        let directory = std::fs::read_to_string(directory_path)
            .expect("failed to read directory")
            .lines()
            .map(|line| {
                let mut parts = line.split_whitespace();
                let embedding_id = parts.next().unwrap().parse::<u32>().unwrap();
                let block_number = parts.next().unwrap().parse::<u64>().unwrap();
                (embedding_id, block_number)
            })
            .collect::<HashMap<_, _>>();

        EmbeddingCache {
            lru: LinkedList::new(),
            node_map: HashMap::new(),
            embeddings: HashMap::new(),
            directory,
            max_size,
        }
    }

    // embedding ids _should_ always be present
    // unless they're not indexed, in which we'd find an io error
    //
    // cloning the embeddings isn't ideal
    // but neither is the borrow checker
    pub fn get(&mut self, embedding_id: u32) -> Result<Box<Embedding>, std::io::Error> {
        let embedding = match self.embeddings.get(&embedding_id).cloned() {
            Some(embedding) => embedding,
            None => {
                let embeddings = self.get_embeddings(embedding_id)?;
                for e in embeddings {
                    if self.lru.len >= self.max_size as usize {
                        let popped = self.lru.pop_back().unwrap();
                        self.embeddings.remove(&popped);
                        self.node_map.remove(&popped);
                    }

                    let id = e.id as u32;
                    if let Some(node) = self.node_map.get(&id) {
                        unsafe {
                            (*node.as_ptr()).detach();
                        }
                    }

                    let new_node = self.lru.push_front(id);
                    self.embeddings.insert(id, e);
                    self.node_map.insert(id, new_node);
                }

                self.embeddings.get(&embedding_id).unwrap().clone()
            }
        };

        let node = self.node_map.get(&embedding_id).unwrap();
        unsafe {
            self.lru.push_front(embedding_id);
            // horrible hack please god refactor this
            self.lru.len -= 1;

            // does this actually deallocate anything lol
            // should have done this in c
            (*node.as_ptr()).detach();
        }

        // TODO: stack + heap allocation? really?
        self.embeddings.entry(embedding_id).and_modify(|e| {
            *e = embedding.clone();
        });

        Ok(Box::new(embedding))
    }

    // loads all embeddings in a block
    // based on a contained embedding id
    // this adds/replaces the bottom k embeddings in the lru
    // if we're at capacity
    fn get_embeddings(&self, embedding_id: u32) -> Result<Vec<Embedding>, std::io::Error> {
        let block_number = match self.directory.get(&embedding_id) {
            Some(block_number) => *block_number,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("embedding {} not found in directory", embedding_id),
                ))
            }
        };

        let filename = format!(
            "{}/{}",
            config::get_data_dir().to_str().unwrap(),
            block_number
        );

        let block = EmbeddingBlock::from_file(&filename, block_number)?;

        Ok(block.embeddings)
    }
}
