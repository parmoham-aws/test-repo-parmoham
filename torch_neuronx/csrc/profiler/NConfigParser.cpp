#include "NConfigParser.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "LogWrapper.h"

namespace torch_neuronx {
namespace profiler {

namespace detail {

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r";
  auto start = s.find_first_not_of(ws);
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

std::vector<std::string> splitAndTrim(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    result.push_back(trim(item));
  }
  return result;
}

std::vector<uint32_t> parseIndexOrRange(const std::string& s) {
  std::vector<uint32_t> result;
  auto dashPos = s.find('-');

  if (dashPos != std::string::npos) {
    // Range format: "start-end"
    std::string startStr = s.substr(0, dashPos);
    std::string endStr = s.substr(dashPos + 1);

    int32_t start = static_cast<int32_t>(std::stol(startStr));
    int32_t end = static_cast<int32_t>(std::stol(endStr));

    if (start > end) {
      NPROF_WARN("  Invalid range (start > end):", s);
      return result;  // Return empty
    }

    for (int32_t i = start; i <= end; ++i) {
      result.push_back(static_cast<uint32_t>(i));
    }
  } else {
    // Single index
    result.push_back(static_cast<uint32_t>(std::stol(s)));
  }

  return result;
}

}  // namespace detail

ConfigValue NConfigParser::parseBool(const std::string& val) {
  std::string lower = val;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "true") {
    return true;
  }
  if (lower == "false") {
    return false;
  }
  throw std::invalid_argument("Unrecognized boolean value: " + val);
}

ConfigValue NConfigParser::parseInt64(const std::string& val) { return std::stoll(val); }

ConfigValue NConfigParser::parseUint64(const std::string& val) {
  return static_cast<uint64_t>(std::stoll(val));
}

ConfigValue NConfigParser::parseString(const std::string& val) { return val; }

ConfigValue NConfigParser::parseNcIndexList(const std::string& val) {
  std::vector<uint32_t> result;
  std::vector<std::string> parts = detail::splitAndTrim(val, ',');

  for (const auto& part : parts) {
    if (!part.empty()) {
      std::vector<uint32_t> indices = detail::parseIndexOrRange(part);
      for (uint32_t idx : indices) {
        result.push_back(idx);
      }
    }
  }

  return result;
}

bool NConfigParser::parseAndApply(const std::string& customConfig, const ConfigRegistry& registry,
                                  void* context) {
  if (customConfig.empty()) {
    return true;
  }

  // Build lookup map from registry
  std::unordered_map<std::string, const ConfigDef*> lookup;
  for (const auto& def : registry) {
    lookup[def.name] = &def;
  }

  // Parse semicolon-delimited pairs
  std::vector<std::string> pairs = detail::splitAndTrim(customConfig, ';');

  for (const auto& pair : pairs) {
    if (pair.empty()) {
      continue;
    }

    std::vector<std::string> kv = detail::splitAndTrim(pair, ':');
    if (kv.size() != 2) {
      NPROF_WARN("  Ignoring malformed option (expected key:value):", pair);
      continue;
    }

    const std::string& name = kv[0];
    const std::string& val = kv[1];

    auto it = lookup.find(name);
    if (it == lookup.end()) {
      NPROF_WARN("  Ignoring unrecognized option:", pair);
      continue;
    }

    try {
      ConfigValue parsed = it->second->parser(val);
      bool ok = it->second->applier(context, parsed);
      if (ok) {
        NPROF_INFO("  Config applied:", name);
      } else {
        NPROF_WARN("  Config applier failed for:", name);
      }
    } catch (const std::exception& e) {
      NPROF_WARN("  Error parsing option:", pair, " (", e.what(), ")");
    }
  }

  return true;
}

}  // namespace profiler
}  // namespace torch_neuronx
