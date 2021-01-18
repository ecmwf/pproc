
#include "cmir.h"

#include <exception>
#include <vector>

#include "eckit/exception/Exceptions.h"
#include "eckit/log/Log.h"

#include "mir/param/SimpleParametrisation.h"

extern "C" {

struct mir_cfg_t {
    mir::param::SimpleParametrisation config;
};

struct mir_cfg_t *mir_cfg_new(void) {
    try {
        return new mir_cfg_t;
    }
    catch (std::exception& e) {
        eckit::Log::error() << "cmir: caught exception: " << e.what() << std::endl;
    }
    catch (...) {
        eckit::Log::error() << "cmir: caught unknown exception" << std::endl;
    }

    return nullptr;
}

int mir_cfg_destroy(struct mir_cfg_t *cfg) {
    try {
        ASSERT(cfg);
        delete cfg;
        return 0;
    }
    catch (std::exception& e) {
        eckit::Log::error() << "cmir: caught exception: " << e.what() << std::endl;
    }
    catch (...) {
        eckit::Log::error() << "cmir: caught unknown exception" << std::endl;
    }

    return 1;
}

#define CFGFUNC(name_, type_) \
    int mir_cfg_set_##name_(struct mir_cfg_t *cfg, const char *name, type_ val) { \
        try { \
            ASSERT(cfg); \
            cfg->config.set(name, val); \
            return 0; \
        } \
        catch (std::exception& e) { \
            eckit::Log::error() << "cmir: caught exception: " << e.what() << std::endl; \
        } \
        catch (...) { \
            eckit::Log::error() << "cmir: caught unknown exception" << std::endl; \
        } \
        return 1; \
    }

CFGFUNC(str, const char *)
CFGFUNC(int, int)
CFGFUNC(long, long)
CFGFUNC(ll, long long)
CFGFUNC(size, size_t)
CFGFUNC(float, float)
CFGFUNC(double, double)

#define CFGFUNCV(name_, type_) \
    int mir_cfg_set_##name_##_v(struct mir_cfg_t *cfg, const char *name, type_ *val, size_t count) { \
        try { \
            ASSERT(cfg); \
            std::vector<type_> v(val, val + count); \
            cfg->config.set(name, v); \
            return 0; \
        } \
        catch (std::exception& e) { \
            eckit::Log::error() << "cmir: caught exception: " << e.what() << std::endl; \
        } \
        catch (...) { \
            eckit::Log::error() << "cmir: caught unknown exception" << std::endl; \
        } \
        return 1; \
    }

CFGFUNCV(int, int)
CFGFUNCV(long, long)
CFGFUNCV(ll, long long)
CFGFUNCV(size, size_t)
CFGFUNCV(float, float)
CFGFUNCV(double, double)

} // extern "C"

