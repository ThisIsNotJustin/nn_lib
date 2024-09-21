#ifndef REGION_H_
#define REGION_H_

#include <stddef.h>

#ifndef REGION_ASSERT
#include <assert.h>
#define REGION_ASSERT assert
#endif

#ifndef REGION_MALLOC
#include <stdlib.h>
#define REGION_MALLOC malloc
#endif

typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *words;
} Region;

#define region_reset(region) (REGION_ASSERT((region) != NULL), (region)->size = 0)
#define region_occupied_bytes(region) (REGION_ASSERT((region) != NULL), (region)->size*sizeof(*(region)->words))
#define region_save(region) (REGION_ASSERT((region) != NULL), (region)->size)
#define region_rewind(region, s) (REGION_ASSERT((region) != NULL), (region)->size = s)

Region region_init(size_t capacity_bytes);
void *region_alloc(Region *region, size_t size_bytes);

#endif // REGION_H_

#ifdef REGION_IMPLEMENTATION

Region region_init(size_t capacity_bytes) {
    Region r = {0};
    size_t word_size = sizeof(*r.words);
    size_t capacity_words = (capacity_bytes + word_size - 1) / word_size;

    void *words = REGION_MALLOC(capacity_words * word_size);
    REGION_ASSERT(words != NULL);
    r.capacity = capacity_words;
    r.words = (uintptr_t *) words;
    return r;
}

void *region_alloc(Region *r, size_t size_bytes) {
    if (r == NULL) return REGION_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1) / word_size;

    REGION_ASSERT(r->size + size_words <= r->capacity);
    if (r->size + size_words > r->capacity) return NULL;
    void *result = &r->words[r->size];
    r->size += size_words;
    return result;
}

#endif // REGION_IMPLEMENTATION

