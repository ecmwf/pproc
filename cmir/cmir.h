#ifndef _CMIR_CMIR_H_
#define _CMIR_CMIR_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mir_cfg_t;

int simple_mir(const char *infile, const char *outfile, struct mir_cfg_t *cfg);


struct mir_cfg_t *mir_cfg_new(void);
int mir_cfg_destroy(struct mir_cfg_t *cfg);

int mir_cfg_set_str(struct mir_cfg_t *cfg, const char *name, const char *val);
int mir_cfg_set_int(struct mir_cfg_t *cfg, const char *name, int val);
int mir_cfg_set_long(struct mir_cfg_t *cfg, const char *name, long val);
int mir_cfg_set_ll(struct mir_cfg_t *cfg, const char *name, long long val);
int mir_cfg_set_size(struct mir_cfg_t *cfg, const char *name, size_t val);
int mir_cfg_set_float(struct mir_cfg_t *cfg, const char *name, float val);
int mir_cfg_set_double(struct mir_cfg_t *cfg, const char *name, double val);

int mir_cfg_set_int_v(struct mir_cfg_t *cfg, const char *name, int *val, size_t count);
int mir_cfg_set_long_v(struct mir_cfg_t *cfg, const char *name, long *val, size_t count);
int mir_cfg_set_ll_v(struct mir_cfg_t *cfg, const char *name, long long *val, size_t count);
int mir_cfg_set_size_v(struct mir_cfg_t *cfg, const char *name, size_t *val, size_t count);
int mir_cfg_set_float_v(struct mir_cfg_t *cfg, const char *name, float *val, size_t count);
int mir_cfg_set_double_v(struct mir_cfg_t *cfg, const char *name, double *val, size_t count);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _CMIR_CMIR_H_
