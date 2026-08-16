/* k3_st.h - safetensors reader adapted for DeepSeek-V4-Flash-0731.
 *
 * Routed expert weights arrive as packed FP4 bytes with E8M0 group scales. The reader
 * also handles the FP8, BF16, integer and metadata dtypes used by the checkpoint.
 *
 * FORMAT, verified against the fixed 48-shard checkpoint revision:
 *   [8 bytes little-endian N] [N bytes of JSON header] [tensor data]
 *   Each header entry: {"dtype": ..., "shape": [...], "data_offsets": [start, end]}
 *   data_offsets are relative to the END of the header, so the absolute file offset
 *   is 8 + N + start.
 *   The data region is fully contiguous with no gaps between tensors. The fixed model
 *   has 48 shards and 67,612 main tensors; optional mtp.* tensors share those shards.
 *
 * WHY A HASH INDEX IS NOT OPTIONAL
 *   Every routed expert load performs six lookups (three matrices, each with a weight
 *   and scale), and every token activates six routed experts in each of 43 layers.
 *   Hash lookup keeps metadata work off the decode critical path.
 *
 * WHY OFFSET-BASED pread
 *   Expert misses copy exactly the selected weight and scale runs into a bounded cache.
 *   Dense layer tensors may use a separate read-only mapping, but streamed experts need
 *   explicit offsets, direct-I/O alignment and deterministic cache ownership.
 */
#ifndef K3_ST_H
#define K3_ST_H

#include <stddef.h>
#include <stdint.h>

/* Dtypes found in the DeepSeek-V4-Flash-0731 checkpoint (verified against the
 * shard headers): BF16 trunk, F32 small vectors, F8_E4M3 FP8 weights with
 * F8_E8M0 per-block scales, I8 packed FP4 expert weights (two nibbles per
 * byte), and I64 routing tables. K3_DT_U8 remains for the synthetic reader fixture. */
typedef enum { K3_DT_UNKNOWN = 0, K3_DT_U8, K3_DT_BF16, K3_DT_F16, K3_DT_F32,
               K3_DT_I8, K3_DT_I64, K3_DT_F8_E4M3, K3_DT_F8_E8M0 } K3Dtype;

typedef struct {
    char     *name;
    int       shard;          /* index into K3St.fd[]                      */
    K3Dtype   dtype;
    int       ndim;
    int64_t   shape[4];
    int64_t   off;            /* ABSOLUTE byte offset within its shard file */
    int64_t   nbytes;
} K3Tensor;

/* O_DIRECT alignment. Offset, length and buffer must all be multiples of this. */
#define K3_ST_ALIGN 4096

typedef struct {
    int       *fd;            /* one open descriptor per shard             */
    int       *dfd;           /* the same shards opened O_DIRECT, or -1    */
    char     **path;
    int        nshard;

    K3Tensor  *t;             /* every tensor, in discovery order          */
    int        nt;

    int32_t   *bucket;        /* open-addressed hash, -1 empty             */
    int        nbucket;

    char      *strpool;       /* all names, one allocation                 */
    size_t     strcap, strlen_;
} K3St;

/* Open every *.safetensors in dir and index every tensor. Returns 0 on success. */
int  k3_st_open(K3St *s, const char *dir);
void k3_st_close(K3St *s);

/* O(1) lookup. Returns NULL when absent, which callers must treat as fatal: a
 * silently missing weight reads as zeros and the model still runs, plausibly wrong. */
const K3Tensor *k3_st_find(const K3St *s, const char *name);

/* Raw bytes, exactly as stored. buf must hold t->nbytes. Returns bytes read. */
int64_t k3_st_read(const K3St *s, const K3Tensor *t, void *buf);

/* Read [off, off+nbytes) from a shard with O_DIRECT, bypassing the page cache.
 *
 * WHY THIS IS NOT JUST k3_st_read WITH A FLAG
 *   O_DIRECT demands that the file offset, the length and the buffer address all be
 *   multiples of K3_ST_ALIGN. An expert run starts wherever the checkpoint put it, so
 *   the read is WIDENED outward to the enclosing aligned window and the caller is told
 *   where the payload actually begins inside buf. buf must therefore hold
 *   nbytes + 2*K3_ST_ALIGN and be page aligned (posix_memalign).
 *
 *   Worth it because a streamed expert is read once and evicted: the page cache can only
 *   copy it twice and push out something useful. Under a 32 GB cgroup cap the buffered
 *   path measured 1,247 MB/s on a disk that does 6,400.
 *
 * Returns payload bytes available, and sets *payload_off. Falls back to a buffered read
 * at offset 0 (setting *payload_off = 0) when O_DIRECT is not available. */
int64_t k3_st_read_aligned(const K3St *s, int shard, int64_t off, int64_t nbytes,
                           void *buf, int64_t bufcap, int64_t *payload_off);

/* Read and widen to float32. Handles F32 (memcpy), BF16 (shift left 16), F16,
 * I64 (value as float), and the byte dtypes U8/I8/F8_E4M3/F8_E8M0 (raw code
 * value as float, for callers that want the quantised codes as numbers).
 * out must hold t->nbytes/elem_size floats. */
int64_t k3_st_read_f32(const K3St *s, const K3Tensor *t, float *out);

/* Elements in a tensor, and bytes per element for its dtype. */
int64_t k3_st_numel(const K3Tensor *t);
int     k3_st_elemsize(K3Dtype d);

/* bf16 -> f32 is a pure bit shift: bf16 IS the top 16 bits of an f32. No table, no
 * rounding, and unlike f16 there is no exponent rebias. */
static inline float k3_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#endif /* K3_ST_H */
