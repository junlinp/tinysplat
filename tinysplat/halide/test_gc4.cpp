#include <Halide.h>
#include <cstdio>
#include <cmath>
#include "algorithm.h"

int main() {
    const int H = 16, W = 32, C = 3;
    
    // Case 1: Buffer declared as {H, W, C} — WRONG (matches old code)
    {
        float buf[H*W*C];
        Halide::Buffer<float> b(buf, {H, W, C});  // dim0=H, dim1=W, dim2=C
        Halide::Func f("f");
        Halide::Var x("x"), y("y"), c("c");
        f(x, y, c) = b(y, x, c);  // Transpose!
        
        // Bounds match buffer dims
        f.bound(x, 0, W);  // x corresponds to dim1=W
        f.bound(y, 0, H);  // y corresponds to dim0=H
        f.bound(c, 0, C);
        
        Halide::Buffer<float> out(W*H*C, {W, H, C});  // output dims match f's vars
        // Realize f — this will try to access f(x=0..W-1, y=0..H-1, c=0..C-1)
        // Which translates to b(y=W-1..0?, x=0..W-1?, c=...) — WRONG ORDER!
        printf("Case 1 (wrong order): skipping realize to avoid crash\n");
    }
    
    // Case 2: Buffer declared as {W, H, C} — CORRECT
    {
        float buf[W*H*C];
        Halide::Buffer<float> b(buf, {W, H, C});  // dim0=W, dim1=H, dim2=C
        Halide::Func f("f");
        Halide::Var x("x"), y("y"), c("c");
        f(x, y, c) = b(y, x, c);  // Transpose: b(y, x, c) accesses b[W*y + x]
        
        // b(y, x, c) with y=0..H-1, x=0..W-1 => b[W*y + x] — valid!
        f.bound(x, 0, W);
        f.bound(y, 0, H);
        f.bound(c, 0, C);
        
        float out[W*H*C];
        Halide::Buffer<float> outbuf(out, {W, H, C});
        f.realize(outbuf);
        printf("Case 2 (correct): realize OK, out[0,0]=%f out[W/2,H/2]=%f\n",
               out[0], out[W*(H/2) + W/2]);
    }
    
    return 0;
}
