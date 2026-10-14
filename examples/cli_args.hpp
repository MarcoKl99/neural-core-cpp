#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace nrt_examples {

/*
Parsed command line for the training examples.

`--log-interval <n>` (or `--log-interval=<n>`) sets how often a row is written to
the CSV (every n epochs); the final epoch is always written. The flag is removed
from the positional arguments, so the examples' existing positional arguments
(CSV path, MNIST subset sizes, ...) keep working regardless of where the flag
appears.

Examples:
    ./xor_deep                                  # default interval
    ./xor_deep my_curve.csv --log-interval 100
    ./mnist_mlp 5000 1000 15 --log-interval=5
*/
struct ParsedArgs {
    std::vector<std::string> positional;  // argv[1..] with the log-interval flag removed
    int log_interval = 1;                 // falls back to the caller's default
};

inline ParsedArgs parse_args(int argc, char** argv, int default_log_interval) {
    ParsedArgs args;
    args.log_interval = default_log_interval;

    const std::string flag = "--log-interval";
    const std::string flag_eq = "--log-interval=";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == flag) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--log-interval requires a value");
            }
            args.log_interval = std::stoi(argv[++i]);
        } else if (arg.rfind(flag_eq, 0) == 0) {
            args.log_interval = std::stoi(arg.substr(flag_eq.size()));
        } else {
            args.positional.push_back(arg);
        }
    }

    if (args.log_interval < 1) {
        throw std::invalid_argument("--log-interval must be >= 1");
    }
    return args;
}

}  // namespace nrt_examples
