//@ charon-args=--mir elaborated
async fn foo(_x: &i32) {}

async fn bar() {
    let mut x = 42;
    if true {
        let y = 0;
        foo(&(x + y)).await;
    }
    x += 1;
}
