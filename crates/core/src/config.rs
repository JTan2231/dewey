#[cfg(debug_assertions)]
const DEBUG: bool = true;
#[cfg(not(debug_assertions))]
const DEBUG: bool = false;

fn create_if_nonexistent(path: &std::path::PathBuf) {
    if !path.exists() {
        match std::fs::create_dir_all(&path) {
            Ok(_) => (),
            Err(e) => panic!("Failed to create directory: {:?}, {}", path, e),
        };
    }
}

fn touch_file(path: &std::path::PathBuf) {
    if !path.exists() {
        match std::fs::File::create(&path) {
            Ok(_) => (),
            Err(e) => panic!("Failed to create file: {:?}, {}", path, e),
        };
    }
}

pub fn get_home_dir() -> std::path::PathBuf {
    match std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .or_else(|_| {
            std::env::var("HOMEDRIVE").and_then(|homedrive| {
                std::env::var("HOMEPATH").map(|homepath| format!("{}{}", homedrive, homepath))
            })
        }) {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => panic!("Failed to get home directory"),
    }
}

pub fn get_config_dir() -> std::path::PathBuf {
    let home_dir = get_home_dir();
    home_dir.join(".config/dewey")
}

pub fn get_local_dir() -> std::path::PathBuf {
    let home_dir = get_home_dir();
    home_dir.join(".local/dewey")
}

pub fn get_data_dir() -> std::path::PathBuf {
    let home_dir = get_home_dir();
    home_dir.join(".local/dewey/data")
}

pub fn setup() {
    let now = match DEBUG {
        true => "debug".to_string(),
        false => chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string(),
    };

    match std::env::var("OPENAI_API_KEY") {
        Ok(_) => (),
        Err(_) => panic!("OPENAI_API_KEY environment variable not set"),
    }

    let config_path = get_config_dir();
    let local_path = get_local_dir();
    let data_path = get_data_dir();

    let logging_path = local_path.join("logs");
    crate::logger::Logger::init(format!(
        "{}/{}.log",
        logging_path.to_str().unwrap(),
        now.clone()
    ));

    create_if_nonexistent(&local_path);
    create_if_nonexistent(&config_path);
    create_if_nonexistent(&logging_path);
    create_if_nonexistent(&data_path);

    touch_file(&local_path.join("ledger"));
    touch_file(&config_path.join("ledger"));
}
