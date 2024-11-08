// TODO: better organization to account for varying meta lengths etc.?

#[macro_export]
macro_rules! create_dir {
    ($path:expr) => {
        match std::fs::create_dir_all($path) {
            Ok(_) => {}
            Err(e) => panic!("Failed to create directory: {:?}, {}", $path, e),
        }
    };
}

#[macro_export]
macro_rules! create_file {
    ($path:expr) => {
        match std::fs::File::create($path) {
            Ok(_) => {}
            Err(e) => panic!("Failed to create file: {:?}, {}", $path, e),
        }
    };
}

#[macro_export]
macro_rules! write_file {
    ($path:expr, $contents:expr) => {
        match std::fs::write($path, $contents) {
            Ok(_) => {}
            Err(e) => panic!("Failed to write to file: {:?}, {}", $path, e),
        }
    };
}

pub struct Cleanup;
impl Drop for Cleanup {
    fn drop(&mut self) {
        std::fs::remove_dir_all(crate::config::get_home_dir()).unwrap()
    }
}

pub fn get_tracked_files() -> Vec<String> {
    vec![
        "a.rs".to_string(),
        "b.rs".to_string(),
        "c.rs".to_string(),
        "src/e.rs".to_string(),
    ]
}

pub fn get_untracked_files() -> Vec<String> {
    vec![
        "d.md".to_string(),
        "src/f.md".to_string(),
        "ignore/g.rs".to_string(),
        "ignore/h.rs".to_string(),
    ]
}

pub fn get_meta() -> Vec<String> {
    vec![
        "rust".to_string(),
        "code".to_string(),
        "tracked".to_string(),
    ]
}

pub fn setup() -> Result<(), std::io::Error> {
    println!("===BEGIN SETUP===");
    crate::config::setup();
    let root = crate::config::get_home_dir();
    let config = crate::config::get_config_dir();

    let target = root.join("test_repo");

    let ledger_contents = vec![format!(
        "{} {}",
        target.to_str().unwrap(),
        get_meta()
            .iter()
            .map(|m| format!("--{}", m))
            .collect::<Vec<String>>()
            .join(" ")
    )]
    .join("\n");

    write_file!(config.join("ledger"), ledger_contents.clone());
    println!("set ledger with:\n{}\n", ledger_contents);

    let rule_contents = vec![
        "* --minlength 128 --maxlength 512 --alphanumeric true",
        "rs --code function",
        "md --split \\n",
    ]
    .join("\n");

    write_file!(config.join("rules"), rule_contents.clone());
    println!("set rules with:\n{}\n", rule_contents);

    create_dir!(target.join(".git"));
    create_file!(target.join(".git").join("whatever"));

    let tracked_files = get_tracked_files();
    let untracked_files = get_untracked_files();

    create_dir!(target.join("src"));
    create_dir!(target.join("ignore"));

    for tf in tracked_files.iter() {
        write_file!(target.join(tf), "a");
    }

    for utf in untracked_files.iter() {
        write_file!(target.join(utf), "b");
    }

    // ignore folder with a variety of files inside

    let gitignore_contents = "*.md\n/ignore";
    write_file!(target.join(".gitignore"), gitignore_contents);
    println!(
        "set {} with:\n{}\n",
        target.join(".gitignore").to_str().unwrap(),
        gitignore_contents
    );

    println!("===END SETUP===\n");

    Ok(())
}
