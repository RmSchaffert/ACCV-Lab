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
 
#include <cuvid_dlopen.h>
#ifdef _WIN32
#include <windows.h>
cuvid_lib cuvid_dlopen(const char* path) { return LoadLibraryA (path); }
void* cuvid_dlsym(cuvid_lib lib, const char* symbol) { return GetProcAddress(lib, symbol); }
char* cuvid_dlerror() { return (char*)"Failed"; }
int cuvid_dlclose(cuvid_lib lib) { return FreeLibrary(lib); }
#endif
