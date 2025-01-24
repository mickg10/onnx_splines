#pragma once

#include <string>
#include <vector>
#include <unordered_map>

class CSVReader {
public:
    // Read CSV file and return columns as vectors of floats
    static std::unordered_map<std::string, std::vector<float>> readCSV(
        const std::string& filename,
        const std::vector<std::string>& columns
    );
};
