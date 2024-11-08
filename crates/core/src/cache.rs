use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::dbio::{get_directory, read_embedding_block, BLOCK_SIZE};
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
    len: usize,
    _boo: PhantomData<&'a T>,
}

pub struct IterMut<'a, T> {
    front: Link<T>,
    len: usize,
    _boo: PhantomData<&'a mut T>,
}

impl<T: Clone> Node<T> {
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
            len: self.len,
            _boo: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            front: self.front,
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
        if self.len > 0 {
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

    // used to keep track of which embeddings are invalid from file updates
    // dirty embeddings are accounted for and removed on cache reads
    dirty_embeddings: HashSet<u32>,

    // ideally this is some multiple of the number of embeddings in a block
    // this _must_ be greater or equal to the number of embeddings in a block
    max_size: u32,
}

impl EmbeddingCache {
    pub fn new(max_size: u32) -> Result<Self, std::io::Error> {
        info!("initializing embedding cache with max size {}", max_size);

        if max_size < BLOCK_SIZE as u32 {
            error!(
                "max_size {} must be greater than or equal to the number of embeddings in a block",
                max_size
            );
            panic!("max_size must be greater than or equal to the number of embeddings in a block");
        }

        let directory = get_directory()?;

        Ok(EmbeddingCache {
            lru: LinkedList::new(),
            node_map: HashMap::new(),
            embeddings: HashMap::new(),
            dirty_embeddings: HashSet::new(),
            directory: directory.id_map,
            max_size,
        })
    }

    fn load_embedding_block(&mut self, embedding_id: u32) -> Result<(), std::io::Error> {
        let block_number = match self.directory.get(&embedding_id) {
            Some(block_number) => *block_number,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("embedding {} not found in directory", embedding_id),
                ))
            }
        };

        let embeddings = read_embedding_block(block_number)?.embeddings;
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
                    self.lru.len -= 1;
                }
            }

            let new_node = self.lru.push_front(id);
            self.embeddings.insert(id, e);
            self.node_map.insert(id, new_node);
        }

        Ok(())
    }

    // embedding ids _should_ always be present
    // unless they're not indexed, in which we'd find an io error
    //
    // embeddings are loaded in blocks
    // querying an embedding that's not currently cached will load the entire block into memory
    //
    // cloning the embeddings isn't ideal
    // but neither is the borrow checker
    pub fn get(&mut self, embedding_id: u32) -> Result<Box<Embedding>, std::io::Error> {
        // fetch the embedding
        let embedding = match self.embeddings.get(&embedding_id).cloned() {
            Some(embedding) => {
                if self.dirty_embeddings.contains(&embedding_id) {
                    self.load_embedding_block(embedding_id)?;
                    self.dirty_embeddings.remove(&embedding_id);

                    self.embeddings.get(&embedding_id).unwrap().clone()
                } else {
                    embedding
                }
            }
            None => {
                self.load_embedding_block(embedding_id)?;
                self.embeddings.get(&embedding_id).unwrap().clone()
            }
        };

        // move the LRU node
        let node = self.node_map.get(&embedding_id).unwrap();
        unsafe {
            let new_node = self.lru.push_front(embedding_id);

            // does this actually deallocate anything lol
            // should have done this in c
            (*node.as_ptr()).detach();
            self.lru.len -= 1;

            self.node_map.insert(embedding_id, new_node);
        }

        // TODO: stack + heap allocation? really?
        self.embeddings.entry(embedding_id).and_modify(|e| {
            *e = embedding.clone();
        });

        Ok(Box::new(embedding))
    }
}
