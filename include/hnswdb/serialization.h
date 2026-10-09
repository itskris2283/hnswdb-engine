#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hnswdb {

class CRC32 {
public:
    CRC32();
    void update(const void* data, std::size_t length);
    uint32_t value() const { return crc_ ^ 0xFFFFFFFF; }
    void reset() { crc_ = 0xFFFFFFFF; }

    static uint32_t compute(const void* data, std::size_t length);

private:
    uint32_t crc_{0xFFFFFFFF};
    static const uint32_t table[256];
};

uint32_t compute_crc32(const uint8_t* data, size_t length);

// Atomic write helper: writes to temp file, flushes/syncs, then renames to target
void atomic_write_file(const std::string& filepath, const std::vector<uint8_t>& buffer);
void read_file_exact(const std::string& filepath, std::vector<uint8_t>& buffer);

} // namespace hnswdb
