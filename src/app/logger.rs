use std::backtrace::Backtrace;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct LoggerState {
    file: Mutex<std::fs::File>,
    path: PathBuf,
    started_at: Instant,
}

static LOGGER: OnceLock<LoggerState> = OnceLock::new();
static PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn log_dir() -> &'static Path {
    Path::new("logs")
}

fn log_path_latest() -> PathBuf {
    log_dir().join("latest.log")
}

fn with_logger_line(level: &str, message: &str) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    let elapsed = logger.started_at.elapsed().as_secs_f32();
    let line = format!(
        "[unix_ms={:>13}][uptime={:>8.3}s][{}] {}\n",
        now_unix_millis(),
        elapsed,
        level,
        message
    );
    match logger.file.lock() {
        Ok(mut file) => {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
        Err(poisoned) => {
            let mut file = poisoned.into_inner();
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
    }
}

pub fn init_logger() -> io::Result<PathBuf> {
    if let Some(logger) = LOGGER.get() {
        return Ok(logger.path.clone());
    }

    fs::create_dir_all(log_dir())?;
    let path = log_path_latest();
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)?;
    let state = LoggerState {
        file: Mutex::new(file),
        path: path.clone(),
        started_at: Instant::now(),
    };
    let _ = LOGGER.set(state);
    with_logger_line("INFO", "logger initialized");
    Ok(path)
}

pub fn install_panic_hook() {
    if PANIC_HOOK_INSTALLED.set(()).is_err() {
        return;
    }
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        let payload = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            *s
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.as_str()
        } else {
            "<non-string panic payload>"
        };
        let location = panic_info
            .location()
            .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
            .unwrap_or_else(|| "unknown_location".to_string());
        with_logger_line("CRASH", &format!("panic at {location}: {payload}"));
        let backtrace = Backtrace::force_capture();
        with_logger_line("CRASH", &format!("backtrace:\n{backtrace}"));
        previous(panic_info);
    }));
}

pub fn log_info(message: impl AsRef<str>) {
    with_logger_line("INFO", message.as_ref());
}

pub fn log_warn(message: impl AsRef<str>) {
    with_logger_line("WARN", message.as_ref());
}
