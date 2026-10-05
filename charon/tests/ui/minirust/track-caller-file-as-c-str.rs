//@ output=run-with-minirust
use std::panic::Location;

unsafe extern "Rust" {
    safe fn minirust_print(value: u32);
}

fn main() {
    let filename = Location::caller().file_as_c_str();
    let nul = unsafe { *filename.as_ptr().add(file!().len()) };
    minirust_print(nul as u32);
}
