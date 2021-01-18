
#include "cmir.h"

#include <exception>
#include <memory>

#include "eckit/exception/Exceptions.h"
#include "eckit/runtime/Main.h"

#include "mir/api/MIRJob.h"
#include "mir/input/MIRInput.h"
#include "mir/input/GribFileInput.h"
#include "mir/output/MIROutput.h"
#include "mir/output/GribFileOutput.h"
#include "mir/param/SimpleParametrisation.h"

using mir::api::MIRJob;
using mir::input::MIRInput;
using mir::output::MIROutput;

extern "C" {

struct mir_cfg_t {
    mir::param::SimpleParametrisation config;
};

int simple_mir(const char *infile, const char *outfile, struct mir_cfg_t *cfg) {
    try {
        static bool initialised = false;
        if (!initialised) {
            const char* argv[2] = {"cmir", 0};
            eckit::Main::initialise(1, const_cast<char**>(argv));
            initialised = true;
        }

        MIRJob job;

        ASSERT(cfg);
        cfg->config.copyValuesTo(job);

        std::unique_ptr<MIRInput> input(new mir::input::GribFileInput(infile));
        ASSERT(input);
        ASSERT(input->next());

        std::unique_ptr<MIROutput> output(new mir::output::GribFileOutput(outfile));
        ASSERT(output);

        job.execute(*input, *output);

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

} // extern "C"

