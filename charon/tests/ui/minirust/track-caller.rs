//@ revisions=poly,run
//@[poly] no-default-options
//@[poly] charon-arg=--start-from=crate::convert
//@[poly] charon-arg=--include=core::convert
//@[poly] output=pretty-llbc
//@[run] output=run-with-minirust

use std::panic::Location;

unsafe extern "Rust" {
    safe fn minirust_print(value: u32);
}

#[track_caller]
fn caller() -> &'static Location<'static> {
    Location::caller()
}

#[track_caller]
fn forward() -> &'static Location<'static> {
    caller()
}

trait Locate {
    #[track_caller]
    fn location(&self) -> &'static Location<'static>;
}

struct Marker;

impl Locate for Marker {
    fn location(&self) -> &'static Location<'static> {
        Location::caller()
    }
}

fn convert(value: bool) -> u32 {
    value.into()
}

fn main() {
    let location = forward();
    minirust_print(location.line());
    minirust_print(location.column());
    let filename = location.file();
    let last_byte = unsafe { *filename.as_ptr().add(filename.len() - 1) };
    minirust_print(last_byte as u32);

    let location = Location::caller();
    minirust_print(location.line());
    minirust_print(location.column());

    let location = Marker.location();
    minirust_print(location.line());
    minirust_print(location.column());

    let marker: &dyn Locate = &Marker;
    let location = marker.location();
    minirust_print(location.line());

    // `Into::into` is not annotated, but its blanket implementation is.
    minirust_print(convert(true));

    let indirect: fn() -> &'static Location<'static> = caller;
    let location = indirect();
    minirust_print(location.line());
}
