#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nrt_examples {

/*
Minimal CSV writer for logging training curves - one row per epoch.

Usage:
    nrt_examples::CsvLogger log("xor_training.csv", "epoch,loss");
    log.row({0.0, 0.25});
    log.row({1.0, 0.09});

Writes to the given path (relative to the current working directory) and
flushes on every row, so the file is always up to date even if a long run is
interrupted.
*/
class CsvLogger {
public:
    CsvLogger(const std::string& path, const std::string& header) : out_(path) {
        if (!out_.is_open()) {
            throw std::runtime_error("CsvLogger: could not open '" + path + "' for writing");
        }
        out_.precision(10);
        out_ << header << '\n';
    }

    void row(const std::vector<double>& values) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) out_ << ',';
            out_ << values[i];
        }
        out_ << '\n';
        out_.flush();
    }

private:
    std::ofstream out_;
};

}  // namespace nrt_examples
