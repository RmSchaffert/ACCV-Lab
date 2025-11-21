/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyNvVideoReader.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

class FixedSizeVideoReaderMap {
   private:
    std::vector<std::string> key_vec;
    std::vector<PyNvVideoReader*> value_vec;
    size_t maxSize;
    size_t nextIdx;

   protected:
    int findStringIndex(const std::string& target) {
        auto it = std::find(this->key_vec.begin(), this->key_vec.end(), target);
        if (it != key_vec.end()) {
            return std::distance(key_vec.begin(), it);
        }
        return -1;  // Return -1 if the string is not found
    }

   public:
    FixedSizeVideoReaderMap(size_t size) : maxSize(size) {
#ifdef IS_DEBUG_BUILD
        std::cout << "Create FixedSizeVideoReaderMap object" << std::endl;
#endif
        nextIdx = 0;
        key_vec.reserve(maxSize);
        value_vec.reserve(maxSize);
    }
    ~FixedSizeVideoReaderMap() {
#ifdef IS_DEBUG_BUILD
        std::cout << "Delete FixedSizeVideoReaderMap object" << std::endl;
#endif
        for (int i = 0; i < maxSize && i < value_vec.size(); i++) {
            if (value_vec[i]) {
                delete value_vec[i];
            }
        }
    }

    int getSize() {
        assert(this->key_vec.size() == this->value_vec.size());
        return this->key_vec.size();
    }

    bool notFull() { return (this->key_vec.size() < this->maxSize); }

    void clearAllReaders() {
        key_vec.clear();
        for (int i = 0; i < maxSize && i < value_vec.size(); i++) {
            if (value_vec[i]) {
                delete value_vec[i];
            }
        }
        value_vec.clear();
        nextIdx = 0;  // Reset the next index since we're clearing everything
    }

    PyNvVideoReader* find(const std::string& key, PyNvVideoReader* value = nullptr) {
        auto index = findStringIndex(key);
        if (index < 0) {
#ifdef IS_DEBUG_BUILD
            std::cout << "Can not find for file: " << key << std::endl;
#endif
            // If the key is not found, we need to add it to the map
            // If the map is full, we need to replace the oldest key
            if (this->key_vec.size() >= this->maxSize) {
                // Replace the oldest entry using circular buffer pattern
                // Replace the oldest key with the new key
                key_vec[nextIdx] = key;
                // Replace the oldest value with the new value
                value_vec[nextIdx]->ReplaceWithFile(key);
#ifdef IS_DEBUG_BUILD
                std::cout << "Replace with file: " << key << std::endl;
#endif
                auto tmpIndex = nextIdx;                  // Get the index of the oldest key
                nextIdx = (nextIdx + 1) % this->maxSize;  // Update the index of the oldest key
                return value_vec[tmpIndex];               // Return the value of the oldest key
            } else {
                // Add new entry
                key_vec.push_back(key);
                value_vec.push_back(value);
                return value_vec.back();
            }
        } else {  // Cache Hit
#ifdef IS_DEBUG_BUILD
            std::cout << "Cache hit for file: " << key << std::endl;
#endif
            return value_vec[index];
        }
    }
};
