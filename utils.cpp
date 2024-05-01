#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <limits.h>
#include <inttypes.h>

static std::string format(const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static int get_key_idx(const gguf_context *ctx, const char *key)
{
    int i = gguf_find_key(ctx, key);
    if (i == -1)
    {
        printf("key %s not found in file\n", key);
    }

    return i;
}

static int32_t get_i32(const gguf_context *ctx, const std::string &key)
{
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_i32(ctx, i);
}

static struct ggml_tensor *get_tensor(struct ggml_context *ctx, const std::string &name)
{
    struct ggml_tensor *cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur)
    {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}