//! Keeps some examples of tricky control-flow to ensure we don't regress the generated code
//! quality.

fn rej_sample(a: &[u8]) -> usize {
    let mut sampled = 0;
    if a[0] < 42 && a[1] < 16 {
        sampled += 100;
    }
    sampled += 101; // This statement is duplicated.
    sampled
}
