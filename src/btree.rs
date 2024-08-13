use crate::openai::Embedding;

#[derive(Debug)]
struct BTree<T> {
    value: T,
    children: Vec<BTree<T>>,
}
