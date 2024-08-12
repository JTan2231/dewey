#[derive(Debug)]
struct BTree<T> {
    value: T,
    children: Vec<BTree<T>>,
}
