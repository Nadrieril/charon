//! Ui tests for the charon compiler. Each `<file>.rs` file in `ui/` will be passed to the charon
//! driver. The corresponding pretty-printed llbc output will be stored in `<file>.llbc`, and CI
//! will ensure these stay up-to-date.
//!
//! Files can start with special comments that affect the test behavior. Supported magic comments:
//! see [`HELP_STRING`].
use anyhow::{anyhow, bail};
use assert_cmd::prelude::{CommandCargoExt, OutputAssertExt};
use indoc::indoc as unindent;
use libtest_mimic::{Outcome, Test};
use std::{
    error::Error,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};
use walkdir::{DirEntry, WalkDir};

use charon_lib::cli_options::{CliOpts, CHARON_ARGS};
use util::{compare_or_overwrite, Action};

mod util;

static TESTS_DIR: &str = "tests/ui";

enum TestKind {
    PrettyLlbc,
    KnownFailure,
    KnownPanic,
    Skip,
}

struct MagicComments {
    test_kind: TestKind,
    /// The options with which to run charon.
    cli_opts: CliOpts,
    /// The options to pass to rustc.
    rustc_opts: Vec<String>,
    /// Whether we should store the test output in a file and check it.
    check_output: bool,
    /// A list of paths to files that must be compiled as dependencies for this test.
    auxiliary_crates: Vec<PathBuf>,
}

static HELP_STRING: &str = unindent!(
    "Options are:
    - `//@ output=pretty-llbc`: record the pretty-printed llbc (default);
    - `//@ known-failure`: a test that is expected to fail.
    - `//@ known-panic`: a test that is expected to panic.
    - `//@ skip`: skip the test.
    
    Other comments can be used to control the behavior of charon:
    - `//@ charon-args=<charon cli options>`
    - `//@ rustc-args=<rustc cli options>`
    - `//@ no-check-output`: don't store the output in a file; useful if the output is unstable or
         differs between debug and release mode.
    - `//@ aux-crate=<file path>`: compile this file as a crate dependency.
    "
);

fn parse_magic_comments(input_path: &std::path::Path) -> anyhow::Result<MagicComments> {
    // Parse the magic comments.
    let mut comments = MagicComments {
        test_kind: TestKind::PrettyLlbc,
        cli_opts: CliOpts::default(),
        rustc_opts: Vec::new(),
        check_output: true,
        auxiliary_crates: Vec::new(),
    };
    for line in read_to_string(input_path)?.lines() {
        let Some(line) = line.strip_prefix("//@") else { break };
        let line = line.trim();
        if line == "known-panic" {
            comments.test_kind = TestKind::KnownPanic;
        } else if line == "known-failure" {
            comments.test_kind = TestKind::KnownFailure;
        } else if line == "output=pretty-llbc" {
            comments.test_kind = TestKind::PrettyLlbc;
        } else if line == "skip" {
            comments.test_kind = TestKind::Skip;
        } else if line == "no-check-output" {
            comments.check_output = false;
        } else if let Some(charon_opts) = line.strip_prefix("charon-args=") {
            use clap::Parser;
            // The first arg is normally the command name.
            let args = ["dummy"].into_iter().chain(charon_opts.split_whitespace());
            comments.cli_opts.update_from(args);
        } else if let Some(rustc_opts) = line.strip_prefix("rustc-args=") {
            comments
                .rustc_opts
                .extend(rustc_opts.split_whitespace().map(|s| s.to_string()));
        } else if let Some(crate_path) = line.strip_prefix("aux-crate=") {
            let crate_path: PathBuf = crate_path.into();
            let crate_path = input_path.parent().unwrap().join(crate_path);
            comments.auxiliary_crates.push(crate_path)
        } else {
            return Err(
                anyhow!("Unknown magic comment: `{line}`. {HELP_STRING}").context(format!(
                    "While processing file {}",
                    input_path.to_string_lossy()
                )),
            );
        }
    }
    Ok(comments)
}

struct Case {
    input_path: PathBuf,
    expected: PathBuf,
    magic_comments: MagicComments,
}

fn setup_test(input_path: PathBuf) -> anyhow::Result<Test<Case>> {
    let name = input_path
        .to_str()
        .unwrap()
        .strip_prefix(TESTS_DIR)
        .unwrap()
        .strip_prefix("/")
        .unwrap()
        .to_owned();
    let expected = input_path.with_extension("out");
    let magic_comments = parse_magic_comments(&input_path)?;
    Ok(Test {
        name: name.clone(),
        kind: "".into(),
        is_ignored: matches!(magic_comments.test_kind, TestKind::Skip),
        is_bench: false,
        data: Case {
            input_path,
            expected,
            magic_comments,
        },
    })
}

fn path_to_crate_name(path: &Path) -> Option<String> {
    Some(
        path.file_name()?
            .to_str()?
            .strip_suffix(".rs")?
            .replace(['-'], "_"),
    )
}

fn perform_test(test_case: &Case, action: Action) -> anyhow::Result<()> {
    // Dependencies
    // Vec of (crate name, path to crate.rs, path to libcrate.rlib).
    let deps: Vec<(String, PathBuf, String)> = test_case
        .magic_comments
        .auxiliary_crates
        .iter()
        .cloned()
        .map(|path| {
            let crate_name = path_to_crate_name(&path).unwrap();
            let rlib_file_name = format!("lib{crate_name}.rlib"); // yep it must start with "lib"
            let rlib_path = path.parent().unwrap().join(rlib_file_name);
            let rlib_path = rlib_path.to_str().unwrap().to_owned();
            (crate_name, path, rlib_path)
        })
        .collect();
    for (crate_name, rs_path, rlib_path) in deps.iter() {
        Command::new("rustc")
            .arg("--crate-type=rlib")
            .arg(format!("--crate-name={crate_name}"))
            .arg("-o")
            .arg(rlib_path)
            .arg(rs_path)
            .output()?
            .assert()
            .try_success()
            .map_err(|e| anyhow!(e.to_string()))?;
    }

    // Call the charon driver.
    let mut options = test_case.magic_comments.cli_opts.clone();
    options.print_llbc = true;
    options.dest_file = Some(test_case.input_path.with_extension("llbc"));
    options.crate_name = Some("test_crate".into());

    let mut cmd = Command::cargo_bin("charon-driver")?;
    cmd.env(CHARON_ARGS, serde_json::to_string(&options).unwrap());
    cmd.arg("rustc");
    cmd.arg(test_case.input_path.to_string_lossy().into_owned());
    cmd.arg("--edition=2021"); // To avoid needing `extern crate`
    for (crate_name, _, rlib_path) in deps {
        cmd.arg(format!("--extern={crate_name}={rlib_path}"));
    }
    for arg in &test_case.magic_comments.rustc_opts {
        cmd.arg(arg);
    }
    let output = cmd.output()?;
    let stderr = String::from_utf8(output.stderr.clone())?;
    let stdout = String::from_utf8(output.stdout.clone())?;

    let test_output = match test_case.magic_comments.test_kind {
        TestKind::KnownPanic => {
            if output.status.code() != Some(101) {
                let status = if output.status.success() {
                    "succeeded"
                } else {
                    "errored"
                };
                bail!("Compilation was expected to panic but instead {status}: {stderr}");
            }
            stderr
        }
        TestKind::KnownFailure => {
            if output.status.success() || output.status.code() == Some(101) {
                let status = if output.status.success() {
                    "succeeded"
                } else {
                    "panicked"
                };
                bail!("Compilation was expected to fail but instead {status}: {stderr}");
            }
            stderr
        }
        TestKind::PrettyLlbc => {
            if !output.status.success() {
                bail!("Compilation failed: {stderr}")
            }
            stdout
        }
        TestKind::Skip => unreachable!(),
    };
    if test_case.magic_comments.check_output {
        compare_or_overwrite(action, test_output, &test_case.expected)?;
    } else {
        // Remove the `out` file if there's one from a previous run.
        if test_case.expected.exists() {
            std::fs::remove_file(&test_case.expected)?;
        }
    }

    Ok(())
}

#[test]
fn ui() -> Result<(), Box<dyn Error>> {
    let action = if std::env::var("IN_CI").as_deref() == Ok("1") {
        Action::Verify
    } else {
        Action::Overwrite
    };

    let root: PathBuf = TESTS_DIR.into();
    let file_filter = |e: &DirEntry| e.file_name().to_str().is_some_and(|s| s.ends_with(".rs"));
    let tests: Vec<_> = WalkDir::new(root)
        .min_depth(1)
        .into_iter()
        .filter_map(|entry| match entry {
            Ok(entry) if !file_filter(&entry) => None,
            res => Some(res),
        })
        .map(|entry| {
            let entry = entry?;
            let test = setup_test(entry.into_path())?;
            anyhow::Result::Ok(test)
        })
        .collect::<anyhow::Result<_>>()?;

    let args = libtest_mimic::Arguments::from_args();
    libtest_mimic::run_tests(&args, tests, move |t| match perform_test(&t.data, action) {
        Ok(()) => Outcome::Passed,
        Err(err) => Outcome::Failed {
            msg: Some(err.to_string()),
        },
    })
    .exit()
}
