#![feature(register_tool)]
#![register_tool(pattern)]
//! Tests for the ml name matcher. This is in the rust test suite so that the llbc file gets
//! generated. Tests on the ml side will then inspect the file and check that each item matches the
//! specified pattern.

mod foo {
    #[pattern::pass("test_crate::foo::bar")]
    #[pattern::fail("crate::foo::bar")]
    #[pattern::fail("foo::bar")]
    fn bar() {}
}

trait Trait<T> {
    fn method<U>();
}

impl<T> Trait<Option<T>> for Box<T> {
    #[pattern::pass(
        "test_crate::{test_crate::Trait<alloc::boxed::Box<@T>, core::option::Option<@T>>}::method"
    )]
    // `Box` is special: it can be abbreviated.
    #[pattern::pass("test_crate::{test_crate::Trait<Box<@T>, core::option::Option<@T>>}::method")]
    // Can't abbreviate `Option`, only `Box` is special like that.
    #[pattern::fail("test_crate::{test_crate::Trait<Box<@T>, Option<@T>>}::method")]
    // More general patterns work too.
    #[pattern::pass(
        "test_crate::{test_crate::Trait<alloc::boxed::Box<@T>, core::option::Option<@U>>}::method"
    )]
    #[pattern::pass("test_crate::{test_crate::Trait<@T, @U>}::method")]
    #[pattern::pass("test_crate::Trait<@T, @U>::method")]
    fn method<U>() {}
}

impl<T, U> Trait<Box<T>> for Option<U> {
    // Using the same variable name twice means they must match. This is not the case here.
    // TODO: this should not pass!
    #[pattern::pass(
        "test_crate::{test_crate::Trait<core::option::Option<@T>, alloc::boxed::Box<@T>>}::method"
    )]
    #[pattern::pass(
        "test_crate::{test_crate::Trait<core::option::Option<@T>, alloc::boxed::Box<@U>>}::method"
    )]
    fn method<V>() {}
}

// `call[i]` is a hack to be able to refer to `Call` statements inside the function body.
#[pattern::pass(call[0], "core::option::{core::option::Option<@T>}::is_some<i32>")]
#[pattern::pass(call[0], "core::option::{core::option::Option<@T>}::is_some<@T>")]
// Generic arguments are required.
#[pattern::fail(call[0], "core::option::{core::option::Option<@T>}::is_some")]
#[pattern::pass(call[1], "ArrayToSliceShared<'_, bool, 1>")]
// #[pattern::pass(call[2], "core::slice::index::{core::ops::index::Index<[@T], @I>}::index<bool, core::ops::range::RangeFrom<usize>>")]
// #[pattern::pass(call[2], "core::ops::index::Index<[@D], @C>::index<bool, @B>")]
#[pattern::pass(call[2], "core::ops::index::Index<[bool], core::ops::range::RangeFrom<usize>>::index")]
fn foo() {
    let _ = Some(0).is_some();
    let slice: &[bool] = &[false];
    let _ = &slice[1..];
}
