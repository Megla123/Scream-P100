# Unified Scream P100

Build the native worker with CUDA 12, a supported host compiler, CMake 3.24 or newer, Git and `patch`. The profile launcher needs Python 3.9 or newer.

```sh
bash build.sh --unified
build-unified/scream-p100 --list-backends
build-unified/scream-p100 --cookbook cookbook.json --recipe inkling-z97 --print-settings
build-unified/scream-p100 --cookbook cookbook.json --recipe inkling-z97
```

`bash run.sh --unified` accepts the same configuration options. Begin with `--backend NAME` to pass arguments straight to a native worker. Use `--configured --backend NAME` to select a backend through the profile launcher and apply the configuration and environment layers.

`--cookbook` and `--recipe` are aliases for `--config` and `--profile`. Use the [profile exchange commands](CONFIGURATION.md#export-and-import-a-profile) to copy a recipe to another cookbook or import it as a child profile.

The build pins Strata v0.2.22 and stages its engine and HTTP adapter in the build directory. Set `SCREAM_STRATA_SOURCE` to a local Git checkout at that revision to use it instead of fetching. The launcher locates the adapter through the path recorded by the build. Set `server_directory` to use another Strata `terminal` directory. The Megla adapter uses the tokenizer and optional vision helper from `bash engine-inkling2/build.sh --helpers-only` and its existing Python environment. Set `server_python` to the Python with the relevant adapter dependencies.

One native executable contains the execution paths below. The HTTP launcher selects the adapter for the worker protocol and keeps its lifecycle and temporary files together.

| Backend | Execution path |
| --- | --- |
| `ink2_v1` | Megla's Inkling 2 Mini tensor-split CUDA path |
| `strata-inkling` | Strata's Inkling loader and its CPU and CUDA paths |
| `strata` | Upstream Strata's Flash-Next path with its experimental Pascal build enabled |
| `strata-split` | Scream's two-card P100 Flash-Next split path |

The optimized kernels and their scheduling stay with their backends. Model formats, batching, draft weights, device placement and vision support still depend on the backend and model. The build preserves the separate entry points selected by the existing build commands. To build the unified worker with Scream's Flash-Next split path and the two Inkling paths, set the CMake option `SCREAM_UNIFIED_UPSTREAM_STRATA=OFF`.

## Stage an isolated runtime

After building, stage the engine and both HTTP adapters under a fresh prefix. Use `bin/scream-p100` inside that prefix, which can be moved as a directory. The benchmark is under `libexec/scream-p100/bench/parity.py`.

```sh
prefix=$(mktemp -d)
cmake --install build-unified --prefix "$prefix" --component scream-runtime
"$prefix/bin/scream-p100" --profile inkling-2-mini-strata --print-settings
```

For Megla HTTP, build and stage the tokenizer, GGUF Python package, and CPU projector. Add the CUDA projector component to the same prefix for a GPU projector profile.

```sh
HOST_CXX=g++-13 bash engine-inkling2/build.sh --helpers-only
cmake --install build-unified --prefix "$prefix" --component scream-megla-helpers
bash engine-inkling2/build-vision-cuda.sh
cmake --install build-unified --prefix "$prefix" --component scream-cuda-projector
```

The target host supplies compatible system and CUDA 12 libraries, an adapter Python environment, and the model files. Point `server_python`, `data_dir`, and the selected profile at the test host's own paths.

## Configuration and profiles

See the [configuration guide](CONFIGURATION.md) for a complete JSON example, the file structure, built-in profile families, inheritance, environment and CLI precedence, and the [settings reference](SETTINGS.md).

## Backend-specific settings

For Strata HTTP serving, start with an existing run JSON in `strata_config` or supply its fields in `strata_settings`. An imported file's `host`, `port`, and `env` are configuration defaults below file-wide settings and profiles. They follow the same configurable precedence against environment values and can be overridden from the CLI. The Strata run settings also provide the tokenizer, working directory and model options. The selected native binary is exposed under the basename expected by the HTTP adapter. Set `mode` to `native` or pass `--mode native` for direct native commands and protocol probes. For example, query Strata Inkling's version with `--profile inkling-2-mini-strata --mode native --set 'engine_options.version=true'`.

Native mode reuses Megla's model and drafter defaults from `data_dir`. For Strata it reuses the imported worker's arguments, environment, and working directory. Strata's saved `user_args` and edits to its generated `args` retain their usual priority, with the resolved unified options applied on top. Relative paths in saved arguments use the configured worker directory. The named `model` setting uses the launch directory. The HTTP adapter applies runtime resource and drafter planning in serve mode.

An imported Flash-Next split run file can provide the layer worker under `split_fallback`. Select `qwen-flash-next-3.8-strata` to start that saved layer worker directly. Automatic fallback also applies when a lazy worker fails to start on its first request. Both workers use the configured `model` and `context` overrides. The fallback keeps its own worker arguments, working directory, and environment defaults. Relative working directories resolve from the launcher in both modes. Shared profile and CLI environment overrides apply to both.

On `strata-split`, set `engine_options.spec` to `0` to skip loading the MTP weights and draft vocabulary. Set `engine_options.lookup` to `0` as well for a single-token decode reference. Positive lookup windows still work without MTP, including with parallel slots. The HTTP adapter also accepts `strata_settings.drafting: "off"` as a default for the split worker when no explicit `--spec` is set.

Inkling defaults to the model's pass count, which is three for the supported Megla model. `num_passes` can select a count up to the trained count. A reduced count changes the model computation and requires disabling model drafters because they use intermediate taps from the full schedule. Megla also requires disabling layer dumps for a shortened schedule. Flash-Next has one pass.

Both Inkling paths use the same layer traversal. Megla allocates KV and ShortConv state for 24, 36, or 48 slots with one, two, or three passes. Its snapshots, checkpoints and parked conversations use the same active slot count. Strata’s portable CUDA path reserves, allocates, and grows KV for active visits. It uses trained slot IDs, with slots 42 through 47 reserved for the final layers, and uses the trained layout for ShortConv.

If CUDA KV growth runs out of device memory partway through, Strata preserves existing KV contents and the prior usable context. Later requests can reuse visits that already grew. If host spill is enabled and needed, each visit maps the remaining tail after its own device capacity.

Strata Inkling's model-backed diagnostics use the configured pass count. To compare prefill and decode across pass counts, use the same model, device placement and context, run `--bench` with the same `--prompt-len` and `--gen`, and set `--draft off` on each run. `--dump-ref` and `--kld-eval` can measure the corresponding change in logits against a reference produced at the trained pass count. The Flash-Next profiles require weights that match the compiled Strata geometry or the Scream two-card path they select.

### Vision projector placement

Set `vision_device` in a profile, `SP100_VISION_DEVICE` in the environment, or `--vision-device` on the command line. Valid explicit values are `cpu` and a full NVIDIA device UUID from `nvidia-smi -L`. Omit the setting or use JSON `null` to let the selected backend apply its usual placement. Only the projector process receives the separate CUDA device mask. The text worker retains its own GPU selection.

Set `vision_residency`, `SP100_VISION_RESIDENCY`, or `--vision-residency` to control the external projector's lifecycle. The shared built-in `default` profile and the engine fallback both use `auto`. A GPU projector loads before HTTP readiness, even with lazy loading of the Strata text model, and stays resident across requests and normal idle periods. CPU projectors keep their existing lifecycle. Use `resident` to preload and retain a projector on either device. Use `on-demand` to start it on the first uncached image and release it after image preparation in Megla or after the configured `vision.elastic_idle_s` idle period in Strata (120 seconds by default). To use an environment value over the default profile, set `--environment-precedence environment` or skip the default profile. The unified setting takes precedence over `vision.elastic` when it selects a lifecycle. An explicit Strata model unload also releases the projector and the next load restores it. The built-in native Strata vision shard follows the text model's lifecycle.

Megla runs the projector on the CPU by default. Supply the projector and helper through `server_options.mmproj` and `server_options.vis`, or use the standard `data_dir` projector with the helper in `engine-inkling2/build`. After `bash engine-inkling2/build.sh --helpers-only`, build the optional CUDA helper with `bash engine-inkling2/build-vision-cuda.sh`. The CUDA build defaults to architectures `60;61` for Pascal GPUs. Set `INK2_VISION_CUDA_ARCHITECTURES` to include the selected card's architecture.

The Megla `env.ink2_vision_timeout_seconds` setting gives each image a deadline that covers waiting for the encoder, starting it, and encoding the image. Set it in the environment with `SP100_ENV__INK2_VISION_TIMEOUT_SECONDS`. The default of `0` leaves the projector deadline disabled.

The Strata Flash-Next paths read projector files and the CUDA-capable reader path from the `vision` section of the Strata configuration. An explicit GPU UUID requires the staged HTTP adapter and a CUDA-capable reader and disables the adapter's CPU fallback.

Strata Inkling uses its native `--vision` shard by default. To use the shared GGUF projector, set `strata_settings.vision.mmproj` to the Inkling vision GGUF and `strata_settings.vision.model` to its text vocabulary GGUF. The native text model can still be a `.strata` file. The staged HTTP launcher uses `ink2_vis` for CPU placement and `ink2_vis_cuda` for a GPU UUID. Set `strata_settings.vision.exe` to override the helper path. The documented helper build and staging commands apply to both Inkling backends.

### Sampling

All profiles can set defaults under `sampling`. The HTTP adapter applies them to fields the client leaves out. In native mode, Flash-Next applies the same block to `GEN` and `BGEN` requests from the worker's stdin. A request can override individual fields. Flash-Next uses up to 64 shortlist entries and a history window of up to 4096 tokens. Imported Strata run-file sampling supplies configuration defaults below the unified file and profiles.

Raw Flash-Next commands accept one quoted `--sampling` argument with the nine fields in the order below. A dash leaves a field at the backend default. Strata layer mode applies the same settings to direct generation, including in speculative verification. Its penalties use the reply's history, starting with the final prompt token. Set `STRATA_PENALTY_PROMPT=1` to include earlier prompt tokens. Legacy sampling options follow the same configuration precedence. For direct layer generation, legacy `--seed` also enables sampling. Setting `sampling.temperature` explicitly selects greedy or sampled generation. Adding other sampling fields alone preserves the existing mode.

Extend `inkling-2-mini-megla-sampling` or set `sampler` to `strata-inkling` on a Megla profile to use the shared Strata Inkling sampler. Set defaults under `sampling`, for example `{"temperature": 0.7, "top_p": 0.9, "top_k": 32}`. Both Inkling paths use these settings for HTTP requests and direct native generation. HTTP and native protocol requests can override the defaults. The block also supports `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `penalty_last_n`, and `seed`. Megla supports `top_k` up to 64 and native Strata Inkling up to 128.

When invoking the binary directly, supply one quoted `--sampling` argument with values in the order `temperature top_p top_k min_p repetition_penalty frequency_penalty presence_penalty penalty_last_n seed`. With Megla, select `--sampler strata-inkling` as well. For native Strata Inkling profiles, `engine_options.temp` is an alias for `sampling.temperature` and follows the same precedence. On the raw binary command line, Strata Inkling's `--temp` overrides `--sampling` for direct token generation.

Megla's native path remains greedy by default. Set `env.ink2_sampling` to `1` to enable upstream's per-request temperature, min-p, and seed. A prepared DFlash 2 GGUF in the standard model directory takes priority over DSpark. Set `engine_options.mtp` to choose a drafter or to null to disable model drafting. The shared sampler accepts one target draw per token and keeps draft tokens that match it. DSpark and prompt lookup still use the CUDA verification window. Megla applies penalties over the generated reply, up to the configured history window.

With DSpark disabled, the staged Strata Inkling adapter preserves the sampling random sequence when a running request moves between the solo path and a batch slot. This also applies while prompt lookup is enabled.

### Inkling prompt lookup

Both Inkling paths can propose tokens from repeated text in the current request and verify them against the model. Megla keeps its existing HTTP `lookup` request field and `env.INK2_OA_LOOKUP` default. For Strata Inkling, set `engine_options.lookup` to an integer from 0 through 7 in the profile or use `--lookup N` with the native `--serve` command. The Strata default is 0. Lookup works with `--draft off` and reduced loop passes. When DSpark is available, lookup takes priority for matches of at least six tokens and DSpark handles shorter matches.

Strata uses the lookup verifier on CPU or CUDA with rollback support when one slot is active. CUDA layer-split profiles must leave lookup at 0. With several active slots Strata continues to batch their decode steps. Lookup verifies each proposal with the same sampler used for ordinary decode.

Both Inkling HTTP adapters accept a per-request `lookup` integer. In Strata Inkling, an omitted field uses the worker's startup `--lookup` value, `0` disables lookup for one request, and positive values up to 7 are capped by the startup setting. The same override can be passed as `lookup=N` in the native generation commands.

### Inkling repetition guard

Set `engine_options.repetition_guard` to `"on"` or `"off"` on either Inkling profile. The native command accepts `--repetition-guard on|off`. Both backends use the same token repetition check in ordinary and speculative decode. With the option omitted, Strata serving uses the guard and Megla leaves it off. Direct generation uses the guard only when explicitly enabled. Megla's HTTP-level reasoning recovery can still trigger if it sees a text loop before the native guard stops generation.

Strata Inkling preserves the guard's reply history when requests move between inference slots. When the native guard ends a text response, both HTTP adapters report `stop`. A tool call interrupted before its closing marker stays incomplete.

### Inkling tool streaming

Set `tool_streaming` to `incremental` in an Inkling profile, pass `--tool-streaming incremental`, or export `SP100_TOOL_STREAMING=incremental`. Both Inkling backends then stream arguments for declared tools using the model's standard name-first JSON envelope. Large arguments arrive as deltas, and the final JSON brace is released only after the native end-of-message token and full envelope validation. Other envelope forms use the adapter's buffered parser. The Strata path requires the HTTP adapter staged by the unified build.

Use `buffered` to disable this incremental path. The default `native` uses the adapter's own behavior. Enable legacy names to use Megla's `INK2_STREAM_TOOL_ARGUMENTS` spelling as an alias for `tool_streaming`.

Run the launcher tests with `python3 -m unittest discover -s unified -v`. The CPU fixtures `inkling_sampling_test`, `inkling_loop_test`, `inkling_lookup_test`, and `native_sampling_fields_test` are available when the unified build is configured with `STRATA_BUILD_TESTS=ON`.

## Baseline comparisons

Use `bench/parity.py` on isolated loopback listeners during an authorized hardware test window. Compare each unified backend against its corresponding standalone build on the same cards and weights. Keep the tokenizer, context, cache, sampling, draft settings, and device placement fixed. Use the trained pass count for parity and run reduced-pass benchmarks separately. For cold prefill measurements, capture each side from a fresh worker. Duplicate a request under another name to measure reuse on that same worker.

Save an ordered array of named OpenAI requests to `workloads.json`. Use the model alias for each listener on the command line. Add representative long prompts, tool-call requests, and image requests for profiles that support them.

```json
[
  {
    "name": "streamed-decode",
    "request": {
      "messages": [{"role": "user", "content": "Explain how B-tree indexes speed up lookups."}],
      "temperature": 0,
      "reasoning_effort": "none",
      "max_tokens": 128,
      "stream": true,
      "stream_options": {"include_usage": true}
    }
  }
]
```

```sh
python3 bench/parity.py capture --url http://127.0.0.1:18080 --model "$BASELINE_MODEL" \
  --requests workloads.json --label standalone --out baseline.json
python3 bench/parity.py capture --url http://127.0.0.1:18081 --model "$CANDIDATE_MODEL" \
  --requests workloads.json --label unified --out unified.json
python3 bench/parity.py compare baseline.json unified.json
```

Run the captures sequentially when the workers need the same GPUs. `--token-file` supplies a bearer credential if the isolated listener requires one. Captures record timings, token counts, and hashes of requests and responses. Comparison checks text, reasoning, reconstructed tool arguments, and finish reasons. Timing ratios require a matching response and matching token and cache counts. Compare native prefill and decode rates with the corresponding standalone backend. Their timing boundaries depend on the backend.

To measure continuous batching, set the same `--concurrency N` for both captures. The capture sends groups of up to N adjacent requests on separate connections and waits for the group to finish before starting the next one. Use distinct prompts within a group and set the worker's parallel capacity for the scenario being measured. Comparison reports each request's latency and each group's elapsed time and total completion tokens per second. The group rate includes queueing, prefill, and decode. A ratio requires matching responses and token and cache counts across the group. Request-level VRAM samples may include allocations for concurrent requests.

Streaming captures record time to response headers and first output, the longest gap between observed SSE events or comments, and the longest gap between pieces of output. Tool calls also record time to the first argument fragment and the number of frames carrying arguments. Keepalive comments count as SSE traffic.

To measure the effect of Inkling prompt lookup, capture the same workload once with `--lookup 0` and again with `--lookup 7`. Keep `lookup` out of the workload JSON. The comparison prints the two overrides alongside response parity, timing, and draft counts. On Strata, start the worker with `--lookup 7` for both captures. Use prompts with the same prefix-cache state for timing comparisons. To measure lookup on its own, also set `--draft off` on the Strata worker or `"mtp": 0` in the Megla workload requests.

On Linux, add `--worker-pid PID` to each capture to sample the isolated launcher or worker and its descendants with `nvidia-smi`. Supply more than one PID by repeating the option. Reports include before and after allocations and sampled peaks by GPU UUID, including a per-request peak when a sample falls inside that request. The default sampling interval is 0.5 seconds. Set the same `--vram-interval` on both sides. Short-lived allocations may occur between samples.

`compare` exits with status 1 for a response mismatch and status 2 for incompatible workloads or an invalid capture. Review the timing values and token counts separately from the response match.
