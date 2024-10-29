# Dewey - A simple document retrieval system

Dewey is a lightweight document retrieval system that uses OpenAI embeddings to find similar documents based on their content.

## Examples

```bash
# Synchronize the ledger with the configuration (i.e. add new files)
dewey -s

# Embed missing items in the ledger
dewey -e

# Reindex embeddings
dewey -r

# Run the Dewey server
dewey_server

# Find similar documents to "How to train a language model"
dewey "How to train a language model"
```

## Installation

1. **Install Rust:**
   - If you haven't already, download and install the Rust toolchain from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install).
2. **Clone the repository:**
   - Clone the Dewey repository: `git clone https://github.com/JoeyDeVries/dewey`
3. **Build the project:**
   - Navigate to the project directory: `cd dewey`
   - Build the project: `cargo build`
4. **Run the Dewey server:**
   - Run the server: `cargo run --features server`
5. **Run the Dewey CLI:**
   - Run the CLI: `cargo run --features cli`

## Usage

### Configuring Dewey

The Dewey configuration is stored in the following directories:

- **`~/.config/dewey`:** Contains the indexing rules and ledger configuration.
- **`~/.local/dewey`:** Contains the index, ledger, and logs.

**Indexing Rules**

The indexing rules file (`~/.config/dewey/rules`) specifies how Dewey should split and filter documents for embedding. Each line represents a rule, with the following format:

```
extension --rule_type value --rule_type value ...
```

**Example:**

```
rs --split "
" --minlength 10
```

This rule applies to `.rs` files and splits them based on newline characters (`
`). It then filters out any chunks that are less than 10 characters long.

**Ledger Configuration**

The ledger configuration file (`~/.config/dewey/ledger`) lists the directories and files that Dewey should index. Each line represents a directory or file, with the following format:

```
path --meta_tag1 --meta_tag2 ...
```

**Example:**

```
/home/joey/rust/dewey/dewey --language "rust" --project "dewey"
```

This entry indexes all files under the `/home/joey/rust/dewey/dewey` directory, adding the tags `language: rust` and `project: dewey` to each file.

### Running the Dewey Server

The Dewey server listens on port `5051` by default. It handles requests from the Dewey CLI, which sends queries and retrieves similar documents.

### Running the Dewey CLI

The Dewey CLI provides several commands for managing Dewey:

- **`dewey -s`:** Synchronizes the ledger with the configuration.
- **`dewey -e`:** Embeds missing items in the ledger.
- **`dewey -r`:** Reindexes embeddings.
- **`dewey [QUERY]`:** Searches for similar documents based on the provided query.

**Options:**

- **`-s`:** Synchronize the ledger with the configuration.
- **`-e`:** Embed missing items in the ledger.
- **`-f`:** Embed all items in the ledger (re-embed).
- **`-r`:** Reindex embeddings.
- **`-b`:** Reblock embeddings.
- **`-h`:** Print this help message.
- **`--filter field,value`:** Filter results based on the provided field and value.

**Examples:**

- **`dewey -s`:** Synchronize the ledger.
- **`dewey -e`:** Embed missing items.
- **`dewey -r`:** Reindex the embeddings.
- **`dewey -s -e`:** Synchronize the ledger and embed missing items.
- **`dewey "How to train a language model" --filter language,rust`:** Search for documents similar to "How to train a language model" that are tagged with "rust".

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
