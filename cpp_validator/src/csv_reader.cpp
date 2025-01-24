#include "csv_reader.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <stdexcept>

std::unordered_map<std::string, std::vector<float>> CSVReader::readCSV(
    const std::string& filename,
    const std::vector<std::string>& columns
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read header
    std::string header_line;
    std::getline(file, header_line);
    std::vector<std::string> headers;
    boost::split(headers, header_line, boost::is_any_of(","));

    // Find column indices
    std::unordered_map<std::string, size_t> column_indices;
    for (const auto& col : columns) {
        auto it = std::find(headers.begin(), headers.end(), col);
        if (it == headers.end()) {
            throw std::runtime_error("Column not found: " + col);
        }
        column_indices[col] = std::distance(headers.begin(), it);
    }

    // Initialize result
    std::unordered_map<std::string, std::vector<float>> result;
    for (const auto& col : columns) {
        result[col] = std::vector<float>();
    }

    // Read data
    std::string line;
    boost::char_separator<char> sep(",");
    while (std::getline(file, line)) {
        boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
        std::vector<std::string> values(tokens.begin(), tokens.end());

        for (const auto& col : columns) {
            size_t idx = column_indices[col];
            if (idx >= values.size()) {
                throw std::runtime_error("Invalid CSV format in file: " + filename);
            }
            result[col].push_back(std::stof(values[idx]));
        }
    }

    return result;
}
