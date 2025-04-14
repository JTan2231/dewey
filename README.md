# Dewey

Dewey is a local LLM embedding index named after the Dewey Decimal System.

## Features

- [x] Embedding plaintext documents
- [x] HNSW index
- [ ] Configurable index choices
- [x] Basic cosine similarity lookup
- [ ] Optimized dot product
- [ ] Embedding non-plaintext documents (e.g., PDFs)
- [ ] Index benchmarking
- [ ] Better testing
- [x] Document splitting
- [x] .gitignore adherence
- [x] Splitting rules (this will probably be ongoing)
- [x] In-memory embedding cache
- [ ] Cache benchmarking
- [ ] Proper documentation

## Usage

Dewey is currently configured to work within the Chamber ecosystem and relies on utilities from [`common`]() for file system setup.
In particular, the `chamber_common::Workspace` must be setup with your project's root directory before Dewey is initialized.

e.g., in [`william`](https://github.com/JTan2231/chamber/blob/master/william/src-tauri/src/lib.rs#L95),
```rust
// Sets up necessary config/local directories and touches required files to keep things from
// breaking/crashing on start up
fn setup() {
    let home_dir = match get_home() {
        Some(d) => d,
        None => {
            panic!("error: $HOME not set");
        }
    };

    let root = if cfg!(dev) {
        format!("{}/.local/william-dev", home_dir)
    } else {
        format!("{}/.local/william", home_dir)
    };

    chamber_common::Workspace::new(&root);

    // Create directories + other initialization...
}

fn main() {
  setup();
  let dewey = match dewey_lib::Dewey::new() {
      Ok(d) => d,
      Err(e) => {
          panic!("Error initializing Dewey: {}", e);
      }
  };

  let results = match dewey.query("my_file.txt", Vec::new(), 10) {
      Ok(ds) => ds,
      Err(e) => {
          panic!("Error fetching references from Dewey: {}; ignoring", e);
      }
  };

  println!("results: {}", results);
}
```
