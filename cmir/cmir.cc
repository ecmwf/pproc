
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

using mir::api::MIRJob;
using mir::input::MIRInput;
using mir::output::MIROutput;

int simple_mir(const char *infile, const char *outfile) {
    try {
        static bool initialised = false;
        if (!initialised) {
            const char* argv[2] = {"cmir", 0};
            eckit::Main::initialise(1, const_cast<char**>(argv));
            initialised = true;
        }

        MIRJob job;

        job.set("grid", "1/1");

        std::unique_ptr<MIRInput> input(new mir::input::GribFileInput(infile));
        ASSERT(input);
        ASSERT(input->next());

        std::unique_ptr<MIROutput> output(new mir::output::GribFileOutput(outfile));
        ASSERT(output);

        job.execute(*input, *output);
    }
    catch (std::exception& e) {
        eckit::Log::error() << "cmir: caught exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        eckit::Log::error() << "cmir: caught unknown exception" << std::endl;
        return 1;
    }

    return 0;
}

