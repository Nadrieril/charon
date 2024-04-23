ifeq (3.81,$(MAKE_VERSION))
  $(error You seem to be using the OSX antiquated Make version. Hint: brew \
    install make, then invoke gmake instead of make)
endif

.PHONY: default
default: build-charon-rust

.PHONY: all
all: build test nix

.PHONY: format
format:
	cd charon && $(MAKE) format
	cd charon-ml && $(MAKE) format

# We use Rust nightly in order to:
# - be able to write a Rustc plugin
# - use Polonius in some tests
# We keep the nightly version in sync in all the crates by copy-pasting
# a template file for the toolchain.
#
# We used to rely on symbolic links (to a file at the root) but it proved
# problematic with Nix (e.g., if you build the Charon crate, you don't have
# access to files in the parent directory).
#
# Rem.: this is not really necessary for the `tests` crate.
.PHONY: generate-rust-toolchain
generate-rust-toolchain: \
	generate-rust-toolchain-charon \
	generate-rust-toolchain-tests \
	generate-rust-toolchain-tests-polonius

.PHONY: generate-rust-toolchain-%
generate-rust-toolchain-%:
	rm -f $*/rust-toolchain
	echo "# This file was automatically generated: if you need to perform modifications," >> $*/rust-toolchain
	echo "# update rust-toolchain.template in the top directory." >> $*/rust-toolchain
	cat rust-toolchain.template >> $*/rust-toolchain

# Build the project in release mode, after formatting the code
.PHONY: build
build: build-charon-rust build-charon-ml

# Build in debug mode, without formatting the code
.PHONY: build-dev
build-dev: build-dev-charon-rust build-dev-charon-ml

.PHONY: build-charon-rust
build-charon-rust: generate-rust-toolchain
	cd charon && $(MAKE)
	mkdir -p bin
	cp -f charon/target/release/charon bin
	cp -f charon/target/release/charon-driver bin

.PHONY: build-dev-charon-rust
build-dev-charon-rust: generate-rust-toolchain
	cd charon && cargo build
	mkdir -p bin
	cp -f charon/target/debug/charon bin
	cp -f charon/target/debug/charon-driver bin

.PHONY: build-charon-ml
build-charon-ml:
	cd charon-ml && $(MAKE)

.PHONY: build-dev-charon-ml
build-dev-charon-ml:
	cd charon-ml && $(MAKE) build-dev

# Build the tests crate, and run the cargo tests
.PHONY: build-tests
build-tests:
	cd tests && $(MAKE) build && $(MAKE) cargo-tests

# Build the tests-polonius crate, and run the cargo tests
.PHONY: build-tests-polonius
build-tests-polonius:
	cd tests-polonius && $(MAKE) build && $(MAKE) cargo-tests

# Build and run the tests
.PHONY: test
test: build-dev charon-tests charon-ml-tests

# Run Charon on various test files
.PHONY: charon-tests
charon-tests: charon-tests-regular charon-tests-polonius
	cd charon && make test

# Run the Charon ML tests on the .ullbc and .llbc files generated by Charon
.PHONY: charon-ml-tests
charon-ml-tests: build-charon-ml charon-tests
	cd charon-ml && make tests

# Run Charon on rustc's ui test suite
.PHONY: rustc-tests
rustc-tests:
	nix build -L '.#rustc-tests'
	@echo "Summary of the results:"
	@cat result/charon-results | cut -d' ' -f 2 | sort | uniq -c

# Prints a summary of the most common test errors.
.PHONY: analyze-rustc-tests
analyze-rustc-tests: rustc-tests
	find result/ -name '*.charon-output' \
		| xargs cat \
		| grep '^error: ' \
		| sed 's/^error: \([^:]*\).*/\1/' \
		| grep -v 'aborting due to .* error' \
		| sort | uniq -c | sort -h

# Run Charon on the files in the tests crate
.PHONY: charon-tests-regular
charon-tests-regular: build-tests
	echo "# Starting the regular tests"
	cd tests && make charon-tests
	echo "# Finished the regular tests"

# Run Charon on the files in the tests-polonius crate
.PHONY: charon-tests-polonius
charon-tests-polonius: build-tests-polonius
	echo "# Starting the Polonius tests"
	cd tests-polonius && make charon-tests
	echo "# Finished the Polonius tests"

.PHONY: clean
clean:
	cd charon/attributes && cargo clean
	cd charon && cargo clean
	cd charon/macros && cargo clean
	cd tests && cargo clean
	cd tests-polonius && cargo clean
	rm -rf tests/ullbc
	rm -rf tests-polonius/ullbc
	rm -rf tests/llbc
	rm -rf tests-polonius/llbc

# Build the Nix packages
.PHONY: nix
nix: nix-tests nix-tests-polonius nix-ml

.PHONY: nix-tests
nix-tests:
	nix build .#checks.x86_64-linux.tests --show-trace -L

.PHONY: nix-tests-polonius
nix-tests-polonius:
	nix build .#checks.x86_64-linux.tests-polonius --show-trace -L

.PHONY: nix-ml
nix-ml:
	nix build .#checks.x86_64-linux.charon-ml-tests --show-trace -L
