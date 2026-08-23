### Hi, I'm Aman 👋

Software engineer building at the systems level — GPU kernels, C++
performance work, and AI-assisted developer tooling.

Currently contributing GPU kernels to open-source ML infrastructure
(llama.cpp) while building RAAG, an architectural analytics platform
that scopes AI refactoring to exactly what a codebase change can reach.

---

### 🔧 Open Source Contributions

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GPU inference engine (30k+ ⭐)
- Implemented a CUDA kernel for 1D pooling (avg/max), closing a backend
  coverage gap. Verified across 216 test cases on 2x NVIDIA T4 GPUs.
  Merged into master by the project's lead maintainer.
  → [PR #27573](https://github.com/ggml-org/llama.cpp/pull/27573)
- Metal backend contribution — causal attention masking op.
  → [PR #27197](https://github.com/ggml-org/llama.cpp/pull/27197) *(in review)*

---

### 🚀 Featured Project — RAAG

**AI-Powered Architectural Analytics Platform** — parses a codebase,
quantifies its coupling/cohesion, and scopes AI refactoring suggestions
to exactly what a change can reach (blast-radius, not text similarity).

- **3.69x parsing speedup** via a `std::jthread` pool over 579 real
  C++ source files (349 → 1290 files/sec)
- **86% test coverage** across 307 tests
- Three-engine, three-language pipeline: C++20 extraction (Tree-sitter) →
  Python dependency-graph analytics (NetworkX) → GraphRAG-scoped retrieval
  and reasoning (Qdrant + Claude)
- CI-gated: blocks a pull request automatically if it pushes a core
  module's instability past threshold
- Ships as a CLI and a [VS Code extension](https://marketplace.visualstudio.com/items?itemName=amankarki151.raag-vscode)

📺 [Demo walkthrough](https://youtu.be/kbh707DNPeU) · 📦 [Repo](https://github.com/amankarki151/RAAG)

**Writing about it:**
- [Building a Parallel C++ Source Parser: jthread, stop_token, and the Deadlock I Didn't See Coming](https://amankarki.hashnode.dev/parallel-cpp-source-parser-jthread-stop-token)
- [I Ran a Coupling Analyzer on nlohmann/json and fmt. It Found a Class Doing 55 Jobs](https://amankarki.hashnode.dev/cpp-coupling-metrics-nlohmann-json-fmt)
- [I Built an AI Refactoring Tool That Can't See More Code Than the Dependency Graph Allows](https://amankarki.hashnode.dev/graphrag-code-refactoring-blast-radius)

---

### 🛠️ Tech I work with

`C++20` `CUDA` `Python` `CMake` `Tree-sitter` `GraphRAG` `Docker`

---

### 📫 Find me

[LinkedIn](https://linkedin.com/in/aman-karki-) · [Hashnode](https://amankarki.hashnode.dev)
