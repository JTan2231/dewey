#[cfg(feature = "cli")]
mod cli;

#[cfg(feature = "server")]
mod server;

fn main() {
    #[cfg(feature = "cli")]
    {
        println!("Running in CLI mode");
        cli::run();
    }

    #[cfg(feature = "server")]
    {
        println!("Running in server mode");
        server::run();
    }

    #[cfg(not(any(feature = "cli", feature = "server")))]
    compile_error!("Either 'cli' or 'server' feature must be enabled");
}
