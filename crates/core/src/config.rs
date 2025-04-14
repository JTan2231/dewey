use chamber_common::{lprint, Logger};

#[cfg(debug_assertions)]
const DEBUG: bool = true;
#[cfg(not(debug_assertions))]
const DEBUG: bool = false;

fn create_if_nonexistent(path: &std::path::PathBuf) {
    if !path.exists() {
        match std::fs::create_dir_all(&path) {
            Ok(_) => (),
            Err(e) => {
                lprint!(error, "Failed to create directory: {:?}, {}", path, e);
                panic!("Failed to create directory: {:?}, {}", path, e);
            }
        };
    }
}

fn touch_file(path: &std::path::PathBuf) {
    if !path.exists() {
        match std::fs::File::create(&path) {
            Ok(_) => (),
            Err(e) => {
                lprint!(error, "Failed to create file: {:?}, {}", path, e);
                panic!("Failed to create file: {:?}, {}", path, e);
            }
        };
    }
}

// setup for Dewey-specific files + directories
// logs for Dewey are redirected to the main client
// this is a _library_, not a standalone program
//
// `chamber_common::Workspace` _must_ be setup before this function is run
// otherwise the `get_*_dir` functions won't be correctly mapped
pub fn setup() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::var("OPENAI_API_KEY") {
        Ok(_) => (),
        Err(e) => {
            lprint!(error, "Dewey OPENAI_API_KEY environment variable not set");
            return Err(Box::new(e));
        }
    }

    let config_path = chamber_common::get_config_dir();
    let local_path = chamber_common::get_local_dir();
    let data_path = chamber_common::get_data_dir();

    create_if_nonexistent(&local_path);
    create_if_nonexistent(&data_path);

    touch_file(&local_path.join("ledger"));
    touch_file(&local_path.join("id_counter"));
    touch_file(&config_path.join("ledger"));
    touch_file(&config_path.join("rules"));

    Ok(())
}
