#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

// Tile configuration structure (from Intel spec)
struct tileconfig {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];   // bytes per row for each tile
    uint8_t  rows[16];    // rows for each tile
};

int main() {
    // Enable AMX at OS level (Linux: PR_GET_X86_AMX_TILE)
    if (prctl(PR_SET_X86_AMX_TILE, 1, 0, 0, 0)) {
        perror("prctl AMX_TILE enable failed");
        return 1;
    }

    // Setup tile configuration
    struct tileconfig tc;
    memset(&tc, 0, sizeof(tc));
    tc.palette_id = 1;    // Palette ID 1 = AMX-BF16
    tc.colsb[0] = 64;     // Tile0: 64 bytes per row
    tc.rows[0]  = 16;     // Tile0: 16 rows
    tc.colsb[1] = 64;     // Tile1
    tc.rows[1]  = 16;
    tc.colsb[2] = 64;     // Accumulator tile
    tc.rows[2]  = 16;

    // Load tile configuration
    _tile_loadconfig(&tc);

    // Example matrices in BF16 format
    __bfloat16 a[16][32] __attribute__((aligned(64)));
    __bfloat16 b[32][16] __attribute__((aligned(64)));
    float c[16][16] __attribute__((aligned(64)));

    // Fill with some values
    for (int i=0; i<16; i++)
        for (int j=0; j<32; j++)
            a[i][j] = (__bfloat16)(i + j);

    for (int i=0; i<32; i++)
        for (int j=0; j<16; j++)
            b[i][j] = (__bfloat16)(i - j);

    memset(c, 0, sizeof(c));

    // Load tiles
    _tile_loadd(0, a, 32 * sizeof(__bfloat16));  // tile0 = A
    _tile_loadd(1, b, 16 * sizeof(__bfloat16));  // tile1 = B
    _tile_zero(2);                               // tile2 = accumulator

    // Compute matrix multiply (BF16 x BF16 -> FP32 accumulate)
    _tile_dpbf16ps(2, 0, 1);

    // Store result
    _tile_stored(2, c, 16 * sizeof(float));

    // Release AMX state
    _tile_release();

    // Print result
    printf("Result matrix C[0][0] = %f\n", c[0][0]);
    printf("Result matrix C[0][1] = %f\n", c[0][1]);

    return 0;
}

