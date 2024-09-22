# Dewey

An embedding index for local files, named after the Dewey decimal system.

# Usage

```bash
git clone https://github.com/JTan2231/dewey.git
cd dewey/dewey
cargo build --release

sudo cp target/release/dewey /usr/bin
sudo cp target/release/dewey_server /usr/bin

# view commands
dewey -h

# once you've added files to ~/.config/dewey/ledger, perform a full embed + index
dewey -rsfb

# server needs to be running to accept queries
# in a separate terminal, run `dewey_server`
# then,
dewey "your query"
```

# Architecture

- Indexed files are added to a ledger file, acting as the opposite of a .gitignore (include these files, instead of ignore)
- Indexed files are re-embedded using GPT whenever they're updated
- An HNSW is used for lookup/indexing
- Basic TCP server on localhost:5050 for lookup, CLI for interaction
