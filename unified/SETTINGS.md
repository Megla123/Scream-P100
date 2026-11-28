# Scream P100 settings reference

This reference is generated from [settings_reference.json](settings_reference.json). The launcher reads the same file when you run `--help` or `--help-settings`. See the [configuration guide](CONFIGURATION.md) for examples and precedence rules.

Worker and adapter options extend this reference through the option maps. The pinned backend defines their additional flags and limits. See the [worker and adapter interfaces](CONFIGURATION.md#backend-option-interfaces).

- [Launch controls and commands](#launch-controls-and-commands)
- [Launcher settings](#launcher-settings)
- [Sampling](#sampling)
- [Worker options](#worker-options)
- [Server options](#server-options)
- [Worker and server environment](#worker-and-server-environment)
- [Strata run configuration](#strata-run-configuration)

## Launch controls and commands

### `config`

Select the user configuration file.

- **Command line** `--config PATH / --cookbook PATH`
- **Environment** `SP100_CONFIG`
- **Valid values** Path to a readable JSON object. Omit it to use shipped profiles and other supplied inputs.
- **Legacy names and notes** SCREAM_CONFIG · --cookbook · cookbook
- **Related settings** `environment_prefix, legacy_names`

### `profile`

Select a named profile and its ancestors.

- **JSON paths** `profile`
- **Command line** `--profile NAME / --recipe NAME`
- **Environment** `SP100_PROFILE`
- **Valid values** A shipped or file-defined profile name. Omit it to apply the default profile when enabled. JSON null selects no explicit profile.
- **Legacy names and notes** SCREAM_PROFILE · --recipe · recipe
- **Related settings** `profiles.<profile>.extends, ignore_default_profile`

### `ignore_default_profile`

Skip automatic inheritance of the default profile.

- **JSON paths** `ignore_default_profile`
- **Command line** `--ignore-default-profile / --ignore-default-recipe`
- **Environment** `SP100_IGNORE_DEFAULT_PROFILE`
- **Valid values** Boolean. Default false. Use --no-ignore-default-profile or --no-ignore-default-recipe to restore automatic inheritance. An explicit selection or extends: default still applies the profile.
- **Legacy names and notes** SCREAM_IGNORE_DEFAULT_PROFILE · --ignore-default-recipe · --no-ignore-default-recipe
- **Related settings** `profile`

### `environment_precedence`

Choose which source wins between JSON and the environment.

- **JSON paths** `environment_precedence`
- **Command line** `--environment-precedence VALUE`
- **Environment** `SP100_ENVIRONMENT_PRECEDENCE`
- **Valid values** config (default) or environment. The command line keeps highest source priority.
- **Legacy names and notes** SCREAM_ENVIRONMENT_PRECEDENCE
- **Related settings** `legacy_name_precedence`

### `environment_prefix`

Set the namespace for launcher inputs.

- **JSON paths** `environment_prefix`
- **Command line** `--environment-prefix VALUE`
- **Environment** `SP100_ENVIRONMENT_PREFIX`
- **Valid values** Uppercase letters, digits, and single underscores, starting with a letter. The built-in prefix is shown in Environment. File discovery uses the bootstrap prefix.
- **Related settings** `config, environment_precedence`

### `legacy_names`

Enable legacy input aliases in configured mode.

- **JSON paths** `legacy_names`
- **Command line** `--legacy-names / --no-legacy-names`
- **Environment** `SP100_LEGACY_NAMES`
- **Valid values** Boolean. Default false. Use --no-legacy-names to override a file or bootstrap environment that enables it.
- **Related settings** `legacy_name_precedence, config`

### `legacy_name_precedence`

Choose the winning spelling when canonical and legacy names coexist in one source.

- **JSON paths** `legacy_name_precedence`
- **Command line** `--legacy-name-precedence VALUE`
- **Environment** `SP100_LEGACY_NAME_PRECEDENCE`
- **Valid values** canonical (default) or legacy. A conflict is reported to standard error with names only.
- **Related settings** `legacy_names, environment_precedence`

### `settings`

Supply file-wide launcher settings.

- **JSON paths** `settings`
- **Valid values** JSON object. It applies beneath profile settings.

### `profiles`

Define or customize named profiles.

- **JSON paths** `profiles`
- **Valid values** JSON object keyed by nonempty profile names. The shipped definitions are in builtin_profiles.json.
- **Related settings** `profile, profiles.<profile>.extends, profiles.<profile>.settings`

### `profiles.<profile>.extends`

Inherit settings from a named parent profile.

- **JSON paths** `profiles.<profile>.extends`
- **Valid values** A known profile name or null. Cycles are invalid. The default profile has no parent. Omission preserves a shipped parent.
- **Related settings** `profile, ignore_default_profile`

### `profiles.<profile>.settings`

Provide the settings that a profile adds or overrides.

- **JSON paths** `profiles.<profile>.settings`
- **Valid values** JSON object with the same keys as settings. A child overrides any ancestor settings that it supplies.
- **Related settings** `settings, profiles.<profile>.extends`

### `export_profile`

Export a profile as versioned JSON.

- **Command line** `--export-profile FILE / --export-recipe FILE`
- **Valid values** Path to a new file, or - for standard output. Resolves the selected file's declarations and profile inheritance without launch overrides.
- **Legacy names and notes** --export-recipe · export_recipe
- **Related settings** `config, profile, import_profile`

### `import_profile`

Import an exported profile into a configuration file.

- **Command line** `--import-profile FILE / --import-recipe FILE`
- **Valid values** Path to version 1 profile JSON. Use --config to select the destination and --profile to give the new entry a different name. An existing profile name is rejected.
- **Legacy names and notes** --import-recipe · import_recipe
- **Related settings** `config, profile, export_profile, parent_profile`

### `parent_profile`

Choose a parent for an imported profile.

- **Command line** `--parent-profile NAME / --parent-recipe NAME`
- **Valid values** An existing shipped or file-defined profile name. Use with --import-profile. The new entry supplies differences against the inherited settings.
- **Legacy names and notes** --parent-recipe · parent_recipe
- **Related settings** `import_profile, profiles.<profile>.extends`

### `help`

Show the canonical launcher options.

- **Command line** `-h, --help`
- **Valid values** Use -h or --help.
- **Related settings** `help_settings`

### `help_settings`

Read the packaged settings reference.

- **Command line** `--help-settings [NAME]`
- **Valid values** Omit the name for the full reference. Supply a canonical name, JSON path, or a supported legacy alias to look up an individual entry.
- **Related settings** `help`

### `print_settings`

Print the resolved launcher settings as JSON and exit.

- **Command line** `--print-settings`
- **Valid values** Boolean command flag. The result can contain configured credentials and precedes adapter device planning and saved tuning.
- **Related settings** `environment_precedence, profile`

### `set`

Override one setting or a member of a settings map.

- **Command line** `--set KEY=JSON`
- **Valid values** Repeat --set KEY=JSON as needed. For an object nested below a first-level map member, pass the member's whole object.
- **Related settings** `settings, engine_options, env, strata_settings`

### `configured`

Force configured dispatch at the native executable.

- **Command line** `--configured`
- **Valid values** Use as the first argument when --backend would otherwise select raw worker dispatch.
- **Related settings** `backend, mode`

### `list_backends`

List the worker implementations compiled into the executable.

- **Command line** `--list-backends`
- **Valid values** Use as the only argument to the native executable.
- **Related settings** `backend`

## Launcher settings

### `backend`

Choose the worker implementation.

- **JSON paths** `settings.backend, profiles.<profile>.settings.backend`
- **Command line** `--backend VALUE`
- **Environment** `SP100_BACKEND`
- **Valid values** ink2_v1, strata-inkling, strata, or strata-split. Availability depends on the build. The internal default is null. A model-specific profile usually selects it.
- **Legacy names and notes** SCREAM_BACKEND
- **Related settings** `profile, mode, model`

### `mode`

Choose HTTP serving or a direct worker invocation.

- **JSON paths** `settings.mode, profiles.<profile>.settings.mode`
- **Command line** `--mode VALUE`
- **Environment** `SP100_MODE`
- **Valid values** serve (default) or native. Server options apply to serve mode.
- **Legacy names and notes** SCREAM_MODE
- **Related settings** `backend, server_options, engine_options`

### `binary`

Select the unified executable used for worker dispatch.

- **JSON paths** `settings.binary, profiles.<profile>.settings.binary`
- **Command line** `--binary VALUE`
- **Environment** `SP100_BINARY`
- **Valid values** Path to an executable. Default is the invoked executable, or the development build path from the Python entry point.
- **Legacy names and notes** SCREAM_BINARY
- **Related settings** `backend`

### `host`

Bind the HTTP service to this address.

- **JSON paths** `settings.host, profiles.<profile>.settings.host`
- **Command line** `--host VALUE`
- **Environment** `SP100_HOST`
- **Valid values** Nonempty listener address. Default 127.0.0.1.
- **Legacy names and notes** SCREAM_HOST
- **Related settings** `port, mode`

### `port`

Bind the HTTP service to this TCP port.

- **JSON paths** `settings.port, profiles.<profile>.settings.port`
- **Command line** `--port VALUE`
- **Environment** `SP100_PORT`
- **Valid values** Integer 0–65535. Default 8080. Port 0 requests an ephemeral test port.
- **Legacy names and notes** SCREAM_PORT
- **Related settings** `host, mode`

### `data_dir`

Locate conventional Megla files.

- **JSON paths** `settings.data_dir, profiles.<profile>.settings.data_dir`
- **Command line** `--data-dir VALUE`
- **Environment** `SP100_DATA_DIR`
- **Valid values** Directory or null (default). Supplies conventional model, drafter, projector, and Python paths when applicable.
- **Legacy names and notes** SCREAM_DATA_DIR
- **Related settings** `model, server_python, vision_device, num_passes`

### `model`

Select the worker's model.

- **JSON paths** `settings.model, profiles.<profile>.settings.model`
- **Command line** `--model VALUE`
- **Environment** `SP100_MODEL`
- **Valid values** Model file or supported model directory, or null (default). Relative paths are resolved from the launch directory. The worker validates the format and architecture.
- **Legacy names and notes** --native · SCREAM_MODEL
- **Related settings** `backend, data_dir, strata_config`

### `context`

Set the inference context length in tokens.

- **JSON paths** `settings.context, profiles.<profile>.settings.context`
- **Command line** `--context VALUE`
- **Environment** `SP100_CONTEXT`
- **Valid values** Integer 1–2147483647 or null (default). Model and worker limits apply. Megla HTTP defaults to 32768 and caps it at 524288.
- **Legacy names and notes** --ctx · --max-context · SCREAM_CONTEXT
- **Related settings** `model, backend`

### `num_passes`

Select the number of times the looped layers are visited.

- **JSON paths** `settings.num_passes, profiles.<profile>.settings.num_passes`
- **Command line** `--num-passes VALUE`
- **Environment** `SP100_NUM_PASSES`
- **Valid values** Positive integer or null (default, trained count). Flash-Next takes one pass. Inkling is capped at its trained count. Reduced Inkling counts require disabling the DSpark drafter and, for Megla, layer dumps.
- **Legacy names and notes** SCREAM_NUM_PASSES
- **Related settings** `backend, engine_options.draft, engine_options.mtp`

### `sampler`

Choose the worker sampler implementation.

- **JSON paths** `settings.sampler, profiles.<profile>.settings.sampler`
- **Command line** `--sampler VALUE`
- **Environment** `SP100_SAMPLER`
- **Valid values** native (default) or strata-inkling. The latter requires an Inkling backend and enables Megla's configurable sampling.
- **Legacy names and notes** SCREAM_SAMPLER · INK2_STRATA_SAMPLER
- **Related settings** `backend, sampling`

### `sampling`

Set default generation sampling fields.

- **JSON paths** `settings.sampling, profiles.<profile>.settings.sampling`
- **Command line** `--sampling JSON`
- **Environment** `SP100_SAMPLING`
- **Valid values** JSON object. Individual fields support null to clear inherited values. HTTP or native protocol requests can override supported fields. Megla requires sampler=strata-inkling.
- **Legacy names and notes** SCREAM_SAMPLING · INK2_SAMPLING_DEFAULTS
- **Related settings** `sampler, sampling.temperature, sampling.seed`

### `tool_streaming`

Choose how an Inkling HTTP adapter delivers tool-call arguments.

- **JSON paths** `settings.tool_streaming, profiles.<profile>.settings.tool_streaming`
- **Command line** `--tool-streaming VALUE`
- **Environment** `SP100_TOOL_STREAMING`
- **Valid values** native (default), incremental, or buffered. Strata incremental delivery requires the staged adapter. Requests must also request streaming.
- **Legacy names and notes** SCREAM_TOOL_STREAMING · INK2_STREAM_TOOL_ARGUMENTS · SCREAM_INKLING_TOOL_STREAM
- **Related settings** `backend, mode`

### `vision_device`

Place the vision projector on the CPU or a particular NVIDIA GPU.

- **JSON paths** `settings.vision_device, profiles.<profile>.settings.vision_device`
- **Command line** `--vision-device VALUE`
- **Environment** `SP100_VISION_DEVICE`
- **Valid values** cpu, a full NVIDIA GPU UUID, or null (default, backend placement). Explicit GPU placement needs a suitable staged helper. Native Strata Inkling accepts precomputed embeddings.
- **Legacy names and notes** SCREAM_VISION_DEVICE · INK2_VISION_DEVICE
- **Related settings** `vision_residency, strata_settings.vision, server_options.mmproj, server_options.vis`

### `vision_residency`

Choose when the external vision projector loads and releases its memory.

- **JSON paths** `settings.vision_residency, profiles.<profile>.settings.vision_residency`
- **Command line** `--vision-residency VALUE`
- **Environment** `SP100_VISION_RESIDENCY`
- **Valid values** auto (shared default profile and fallback): load a GPU projector before HTTP readiness and keep it resident. Keep the backend's existing CPU lifecycle. resident: apply eager loading and residency to either device. on-demand: load for uncached images and release afterward (after vision.elastic_idle_s in Strata, default 120 seconds). Applies to HTTP projectors. A Strata model unload requested through the API releases the projector too.
- **Legacy names and notes** SCREAM_VISION_RESIDENCY · INK2_VISION_RESIDENCY
- **Related settings** `vision_device, strata_settings.vision, mode`

### `server_directory`

Override the HTTP adapter directory.

- **JSON paths** `settings.server_directory, profiles.<profile>.settings.server_directory`
- **Command line** `--server-directory VALUE`
- **Environment** `SP100_SERVER_DIRECTORY`
- **Valid values** Directory or null (default, packaged or recorded adapter path).
- **Legacy names and notes** --strata · SCREAM_SERVER_DIRECTORY
- **Related settings** `server_python, mode`

### `server_python`

Choose the HTTP adapter Python executable.

- **JSON paths** `settings.server_python, profiles.<profile>.settings.server_python`
- **Command line** `--server-python VALUE`
- **Environment** `SP100_SERVER_PYTHON`
- **Valid values** Executable path or null (default). Megla uses the data-directory virtual environment if supplied. Strata uses the launcher Python by default.
- **Legacy names and notes** SCREAM_SERVER_PYTHON
- **Related settings** `server_directory`

### `strata_config`

Import a saved Strata run configuration.

- **JSON paths** `settings.strata_config, profiles.<profile>.settings.strata_config`
- **Command line** `--strata-config VALUE`
- **Environment** `SP100_STRATA_CONFIG`
- **Valid values** Readable JSON path or null (default). Supplies the saved worker configuration and lower-priority host, port, environment, and sampling defaults.
- **Legacy names and notes** SCREAM_STRATA_CONFIG
- **Related settings** `strata_settings, backend`

### `engine_options`

Pass component-specific options to the worker.

- **JSON paths** `settings.engine_options, profiles.<profile>.settings.engine_options`
- **Command line** `--engine-options JSON`
- **Environment** `SP100_ENGINE_OPTIONS`
- **Valid values** JSON object keyed by the snake_case form of the native flags. A scalar supplies one operand, true emits a bare flag, false or null removes it, a list supplies operands, and a list of lists repeats it.
- **Legacy names and notes** SCREAM_ENGINE_OPTIONS
- **Related settings** `engine_options.<option>, backend, server_options`

### `server_options`

Pass component-specific options to the HTTP adapter.

- **JSON paths** `settings.server_options, profiles.<profile>.settings.server_options`
- **Command line** `--server-options JSON`
- **Environment** `SP100_SERVER_OPTIONS`
- **Valid values** JSON object using the same flag-value shape as engine_options. The launcher owns the listener and backend connection settings.
- **Legacy names and notes** SCREAM_SERVER_OPTIONS
- **Related settings** `server_options.<option>, mode, host, port`

### `env`

Set environment variables forwarded to the worker or adapter.

- **JSON paths** `settings.env, profiles.<profile>.settings.env`
- **Command line** `--env JSON`
- **Environment** `SP100_ENV`
- **Valid values** JSON object keyed by lowercase environment names. String, number, or boolean becomes text. Boolean becomes 1 or 0. Null clears a variable when it wins. Service-manager and launcher-owned variables can be reserved.
- **Legacy names and notes** SCREAM_ENV
- **Related settings** `env.<variable>, environment_precedence`

### `strata_settings`

Customize fields in the Strata adapter run configuration.

- **JSON paths** `settings.strata_settings, profiles.<profile>.settings.strata_settings`
- **Command line** `--strata-settings JSON`
- **Environment** `SP100_STRATA_SETTINGS`
- **Valid values** JSON object merged by key. Nested objects merge and arrays replace. exe, args, env, host, and port are launcher-owned in this map.
- **Legacy names and notes** SCREAM_STRATA_SETTINGS
- **Related settings** `strata_settings.<field>, strata_config, server_options`

## Sampling

### `sampling.temperature`

Scale the next-token distribution; 0 selects greedy decoding.

- **JSON paths** `settings.sampling.temperature, profiles.<profile>.settings.sampling.temperature`
- **Command line** `--set 'sampling.temperature=JSON'`
- **Environment** `SP100_SAMPLING__TEMPERATURE`
- **Valid values** Finite value from 0 through the maximum 32-bit float. Zero selects greedy decoding. Strata's live shared-settings endpoint accepts 0–2. Null clears the configured field.
- **Legacy names and notes** --temp · --temperature · --greedy · Engine flags --temp (Strata Inkling native) and --temperature or --greedy (Strata layer native).
- **Related settings** `sampling.top_p, sampling.top_k`

### `sampling.top_p`

Retain the highest-probability candidates covering this cumulative probability.

- **JSON paths** `settings.sampling.top_p, profiles.<profile>.settings.sampling.top_p`
- **Command line** `--set 'sampling.top_p=JSON'`
- **Environment** `SP100_SAMPLING__TOP_P`
- **Valid values** Finite fraction greater than 0 and at most 1, or null.
- **Legacy names and notes** Strata layer native --top-p.
- **Related settings** `sampling.top_k, sampling.min_p`

### `sampling.top_k`

Limit the maximum number of candidate tokens.

- **JSON paths** `settings.sampling.top_k, profiles.<profile>.settings.sampling.top_k`
- **Command line** `--set 'sampling.top_k=JSON'`
- **Environment** `SP100_SAMPLING__TOP_K`
- **Valid values** Integer 1–64 for Megla and Flash-Next or up to 128 for native Strata Inkling. Strata HTTP startup defaults accept 1–64. Null clears it.
- **Legacy names and notes** Strata layer native --top-k. The native legacy flag can treat 0 as the full shortlist.
- **Related settings** `sampling.temperature, sampling.top_p`

### `sampling.min_p`

Drop candidates below this fraction of the most likely candidate's probability.

- **JSON paths** `settings.sampling.min_p, profiles.<profile>.settings.sampling.min_p`
- **Command line** `--set 'sampling.min_p=JSON'`
- **Environment** `SP100_SAMPLING__MIN_P`
- **Valid values** Finite fraction 0–1, or null. Zero disables the filter.
- **Related settings** `sampling.top_p`

### `sampling.repetition_penalty`

Rescale logits for tokens already present in the penalty history.

- **JSON paths** `settings.sampling.repetition_penalty, profiles.<profile>.settings.sampling.repetition_penalty`
- **Command line** `--set 'sampling.repetition_penalty=JSON'`
- **Environment** `SP100_SAMPLING__REPETITION_PENALTY`
- **Valid values** Finite value greater than 0, or null. 1 leaves logits unchanged.
- **Legacy names and notes** repeat_penalty in an imported Strata sampling block.
- **Related settings** `sampling.penalty_last_n`

### `sampling.frequency_penalty`

Adjust a token's score in proportion to its prior occurrences.

- **JSON paths** `settings.sampling.frequency_penalty, profiles.<profile>.settings.sampling.frequency_penalty`
- **Command line** `--set 'sampling.frequency_penalty=JSON'`
- **Environment** `SP100_SAMPLING__FREQUENCY_PENALTY`
- **Valid values** Finite value -2–2 for the shared native sampler and unified Flash-Next. Other Strata adapter startup validators may require a nonnegative value. Null clears it.
- **Related settings** `sampling.penalty_last_n, sampling.presence_penalty`

### `sampling.presence_penalty`

Adjust a token's score once if it occurs in the penalty history.

- **JSON paths** `settings.sampling.presence_penalty, profiles.<profile>.settings.sampling.presence_penalty`
- **Command line** `--set 'sampling.presence_penalty=JSON'`
- **Environment** `SP100_SAMPLING__PRESENCE_PENALTY`
- **Valid values** Finite value -2–2 for the shared native sampler and unified Flash-Next. Other Strata adapter startup validators may require a nonnegative value. Null clears it.
- **Related settings** `sampling.penalty_last_n, sampling.frequency_penalty`

### `sampling.penalty_last_n`

Set the history window for the three token penalties.

- **JSON paths** `settings.sampling.penalty_last_n, profiles.<profile>.settings.sampling.penalty_last_n`
- **Command line** `--set 'sampling.penalty_last_n=JSON'`
- **Environment** `SP100_SAMPLING__PENALTY_LAST_N`
- **Valid values** Nonnegative integer or null. Flash-Next native supports up to 4096 tokens. Backend protocol defaults apply when the value is omitted.
- **Legacy names and notes** repeat_last_n in an imported Strata sampling block.
- **Related settings** `sampling.repetition_penalty, sampling.frequency_penalty, sampling.presence_penalty`

### `sampling.seed`

Set the default seed for sampled generation.

- **JSON paths** `settings.sampling.seed, profiles.<profile>.settings.sampling.seed`
- **Command line** `--set 'sampling.seed=JSON'`
- **Environment** `SP100_SAMPLING__SEED`
- **Valid values** Unsigned 64-bit integer or null. Unified Flash accepts zero at startup. Strata's shared settings endpoint requires a positive seed.
- **Legacy names and notes** Strata layer native --seed also enables sampling.
- **Related settings** `sampling.temperature`

## Worker options

### `engine_options.<option>`

Configure any option supported by the selected worker.

- **JSON paths** `settings.engine_options.<option>, profiles.<profile>.settings.engine_options.<option>`
- **Command line** `--set 'engine_options.<option>=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__<OPTION>`
- **Valid values** Use the worker flag's snake_case name and its documented operands. Worker options and ranges are specific to the backend and build. Use raw backend help or the pinned source for the complete native interface.
- **Related settings** `engine_options, backend`

### `engine_options.parallel`

Choose the maximum number of parallel Strata Inkling request slots.

- **JSON paths** `settings.engine_options.parallel, profiles.<profile>.settings.engine_options.parallel`
- **Command line** `--set 'engine_options.parallel=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__PARALLEL`
- **Valid values** Integer 1–64 on Strata Inkling. Other backends have their own batching controls.
- **Legacy names and notes** --parallel · --batch · --slots · --slots on Strata Inkling. --batch is also equivalent on that backend.
- **Related settings** `engine_options.batch`

### `engine_options.batch`

Set the worker's native batch slot count.

- **JSON paths** `settings.engine_options.batch, profiles.<profile>.settings.engine_options.batch`
- **Command line** `--set 'engine_options.batch=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__BATCH`
- **Valid values** Positive integer in the worker's supported range. On Strata Inkling use parallel for the same setting.
- **Legacy names and notes** --batch
- **Related settings** `engine_options.parallel`

### `engine_options.kv`

Choose the worker's KV-cache representation.

- **JSON paths** `settings.engine_options.kv, profiles.<profile>.settings.engine_options.kv`
- **Command line** `--set 'engine_options.kv=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__KV`
- **Valid values** Strata Inkling accepts auto, fp16, fp8, k8v8, and k8v4. f16, fp8_e4m3, i8, and q8 are accepted aliases on that backend. Other workers have their own types.
- **Legacy names and notes** --kv · --kv-dtype · --kv-dtype on Strata Inkling.
- **Related settings** `context`

### `engine_options.gpus`

Select GPU devices for the native Strata Inkling worker.

- **JSON paths** `settings.engine_options.gpus, profiles.<profile>.settings.engine_options.gpus`
- **Command line** `--set 'engine_options.gpus=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__GPUS`
- **Valid values** On Strata Inkling, use a GPU count or a comma-separated list of device ordinals. An explicit list puts the primary card first. Use devices when passing a single ordinal.
- **Legacy names and notes** --gpus · IK2_GPUS · --devices on Strata Inkling.
- **Related settings** `env.cuda_visible_devices, vision_device, engine_options.ep_split`

### `engine_options.ep_split`

Set the two-GPU Strata Inkling expert placement ratio.

- **JSON paths** `settings.engine_options.ep_split, profiles.<profile>.settings.engine_options.ep_split`
- **Command line** `--set 'engine_options.ep_split=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__EP_SPLIT`
- **Valid values** Fraction greater than 0 and less than 1. The default is 0.5. It sets the second GPU's share on the applicable expert-parallel path.
- **Legacy names and notes** --ep-split · IK2_EP_SPLIT
- **Related settings** `engine_options.gpus`

### `engine_options.rope_scale`

Set the Flash-Next split worker's rotary position scale.

- **JSON paths** `settings.engine_options.rope_scale, profiles.<profile>.settings.engine_options.rope_scale`
- **Command line** `--set 'engine_options.rope_scale=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__ROPE_SCALE`
- **Valid values** Finite number. 1 disables scaling. Values greater than 1 enable scaling. Omission uses the model's default rotary position behavior.
- **Legacy names and notes** --rope-scale · FNX_ROPE_SCALE
- **Related settings** `context`

### `engine_options.draft`

Choose the Strata Inkling drafter.

- **JSON paths** `settings.engine_options.draft, profiles.<profile>.settings.engine_options.draft`
- **Command line** `--set 'engine_options.draft=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DRAFT`
- **Valid values** off or dspark in the integrated Strata Inkling path. DSpark needs compatible weights and the trained loop-pass count.
- **Legacy names and notes** --draft
- **Related settings** `num_passes, engine_options.draft_beam`

### `engine_options.draft_beam`

Set the Strata Inkling DSpark draft beam width.

- **JSON paths** `settings.engine_options.draft_beam, profiles.<profile>.settings.engine_options.draft_beam`
- **Command line** `--set 'engine_options.draft_beam=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DRAFT_BEAM`
- **Valid values** Integer 1–8 on the staged adapter's DSpark planner. Requires compatible drafting.
- **Legacy names and notes** --draft-beam · STRATA_INK_SPEC_BEAM
- **Related settings** `engine_options.draft`

### `engine_options.draft_cost`

Set the Strata Inkling draft search cost parameter.

- **JSON paths** `settings.engine_options.draft_cost, profiles.<profile>.settings.engine_options.draft_cost`
- **Command line** `--set 'engine_options.draft_cost=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DRAFT_COST`
- **Valid values** Finite numeric share of the verify-step cost model. The Strata Inkling backend default is 4.2.
- **Legacy names and notes** --draft-cost · STRATA_INK_SPEC_COST
- **Related settings** `engine_options.draft`

### `engine_options.draft_plain`

Choose the Strata Inkling plain DSpark path.

- **JSON paths** `settings.engine_options.draft_plain, profiles.<profile>.settings.engine_options.draft_plain`
- **Command line** `--set 'engine_options.draft_plain=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DRAFT_PLAIN`
- **Valid values** Numeric threshold. A positive value enables fallback to a plain token when verification is too expensive. A value of 1.0 represents break-even.
- **Legacy names and notes** --draft-plain · STRATA_INK_SPEC_PLAIN
- **Related settings** `engine_options.draft`

### `engine_options.mtp`

Supply the path to Megla DSpark or Flash-Next MTP draft weights.

- **JSON paths** `settings.engine_options.mtp, profiles.<profile>.settings.engine_options.mtp`
- **Command line** `--set 'engine_options.mtp=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__MTP`
- **Valid values** Megla takes a compatible DSpark file. The Flash-Next split worker takes its MTP runtime directory, required when spec is above zero. Null clears an inherited flag.
- **Legacy names and notes** --mtp
- **Related settings** `data_dir, num_passes, engine_options.spec`

### `engine_options.spec`

Set speculative draft depth for Flash-Next.

- **JSON paths** `settings.engine_options.spec, profiles.<profile>.settings.engine_options.spec`
- **Command line** `--set 'engine_options.spec=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__SPEC`
- **Valid values** Nonnegative integer supported by the worker. Zero disables MTP drafting on the split worker.
- **Legacy names and notes** --spec
- **Related settings** `engine_options.mtp, engine_options.lookup`

### `engine_options.lookup`

Set the prompt-lookup draft window.

- **JSON paths** `settings.engine_options.lookup, profiles.<profile>.settings.engine_options.lookup`
- **Command line** `--set 'engine_options.lookup=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__LOOKUP`
- **Valid values** Strata Inkling permits 0–7, with zero disabling it. The Flash-Next split worker also supports disabling prompt lookup with zero.
- **Legacy names and notes** --lookup
- **Related settings** `engine_options.draft, engine_options.spec, env.ink2_oa_lookup`

### `engine_options.repetition_guard`

Enable or disable the Inkling native repetition check.

- **JSON paths** `settings.engine_options.repetition_guard, profiles.<profile>.settings.engine_options.repetition_guard`
- **Command line** `--set 'engine_options.repetition_guard=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__REPETITION_GUARD`
- **Valid values** on or off. Omission uses the backend's mode default. Strata serving enables it by default. Megla leaves it off by default.
- **Legacy names and notes** --repetition-guard · --no-anti-loop · --no-anti-loop on Strata Inkling disables the guard.
- **Related settings** `tool_streaming`

### `engine_options.devices`

Select one or more explicit Strata Inkling device ordinals.

- **JSON paths** `settings.engine_options.devices, profiles.<profile>.settings.engine_options.devices`
- **Command line** `--set 'engine_options.devices=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DEVICES`
- **Valid values** One ordinal or a comma-separated list in the worker's CUDA visibility. The first is primary.
- **Legacy names and notes** --devices · IK2_GPUS
- **Related settings** `engine_options.gpus, env.cuda_visible_devices`

### `engine_options.layer_split`

Control which layers are assigned to each GPU in a layer-split worker.

- **JSON paths** `settings.engine_options.layer_split, profiles.<profile>.settings.engine_options.layer_split`
- **Command line** `--set 'engine_options.layer_split=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__LAYER_SPLIT`
- **Valid values** auto or a worker-supported list of starting layers. The Flash-Next split worker splits each layer across its cards and does not use the operand.
- **Legacy names and notes** --layer-split
- **Related settings** `engine_options.gpus, strata_settings.layer_split`

### `engine_options.vram_reserve_mib`

Reserve GPU memory for other applications or helpers.

- **JSON paths** `settings.engine_options.vram_reserve_mib, profiles.<profile>.settings.engine_options.vram_reserve_mib`
- **Command line** `--set 'engine_options.vram_reserve_mib=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__VRAM_RESERVE_MIB`
- **Valid values** Nonnegative integer in MiB on supported workers. Hardware planning and worker defaults apply when omitted.
- **Legacy names and notes** --vram-reserve-mib
- **Related settings** `vision_device, server_options.min_free_vram_mib`

### `engine_options.kv_offload`

Select the Strata Inkling KV offload policy.

- **JSON paths** `settings.engine_options.kv_offload, profiles.<profile>.settings.engine_options.kv_offload`
- **Command line** `--set 'engine_options.kv_offload=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__KV_OFFLOAD`
- **Valid values** auto, on, or off. Worker capabilities and selected compute platform apply.
- **Legacy names and notes** --kv-offload
- **Related settings** `engine_options.kv, context`

### `engine_options.draft_vocab`

Select the vocabulary remapping file used by a draft worker.

- **JSON paths** `settings.engine_options.draft_vocab, profiles.<profile>.settings.engine_options.draft_vocab`
- **Command line** `--set 'engine_options.draft_vocab=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DRAFT_VOCAB`
- **Valid values** Readable compatible vocabulary file. Megla can supply its bundled mapping for a conventional launch.
- **Legacy names and notes** --draft-vocab
- **Related settings** `engine_options.mtp`

### `engine_options.dspark`

Point Strata Inkling at DSpark weights.

- **JSON paths** `settings.engine_options.dspark, profiles.<profile>.settings.engine_options.dspark`
- **Command line** `--set 'engine_options.dspark=JSON'`
- **Environment** `SP100_ENGINE_OPTIONS__DSPARK`
- **Valid values** Readable compatible drafter path. Requires matching Inkling weights and trained loop-pass count.
- **Legacy names and notes** --dspark
- **Related settings** `engine_options.draft, num_passes`

## Server options

### `server_options.<option>`

Configure any option supported by the selected HTTP adapter.

- **JSON paths** `settings.server_options.<option>, profiles.<profile>.settings.server_options.<option>`
- **Command line** `--set 'server_options.<option>=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__<OPTION>`
- **Valid values** Use the adapter flag's snake_case name and its documented operands. Launcher-owned flags cannot be replaced. Use the adapter's help or source for additional options.
- **Related settings** `server_options, server_directory`

### `server_options.lazy`

Start the Strata listener before loading the worker.

- **JSON paths** `settings.server_options.lazy, profiles.<profile>.settings.server_options.lazy`
- **Command line** `--set 'server_options.lazy=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__LAZY`
- **Valid values** Boolean flag. true defers the first load to an inference or explicit load request.
- **Legacy names and notes** --lazy
- **Related settings** `strata_settings.lazy_load, mode`

### `server_options.mmproj`

Select a Megla image projector file.

- **JSON paths** `settings.server_options.mmproj, profiles.<profile>.settings.server_options.mmproj`
- **Command line** `--set 'server_options.mmproj=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__MMPROJ`
- **Valid values** Readable projector model path. Requires the companion helper and a vision-enabled launch.
- **Legacy names and notes** --mmproj
- **Related settings** `server_options.vis, vision_device`

### `server_options.vis`

Select a Megla projector helper executable.

- **JSON paths** `settings.server_options.vis, profiles.<profile>.settings.server_options.vis`
- **Command line** `--set 'server_options.vis=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__VIS`
- **Valid values** Executable path. Must match the requested CPU or CUDA placement.
- **Legacy names and notes** --vis
- **Related settings** `server_options.mmproj, vision_device`

### `server_options.alias`

Set the model identifier advertised by the Megla HTTP adapter.

- **JSON paths** `settings.server_options.alias, profiles.<profile>.settings.server_options.alias`
- **Command line** `--set 'server_options.alias=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__ALIAS`
- **Valid values** Nonempty identifier. Managed clients must request the exact registered model alias.
- **Legacy names and notes** --alias
- **Related settings** `model, mode`

### `server_options.max_queue`

Limit the Strata HTTP request queue.

- **JSON paths** `settings.server_options.max_queue, profiles.<profile>.settings.server_options.max_queue`
- **Command line** `--set 'server_options.max_queue=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__MAX_QUEUE`
- **Valid values** Integer from 0 through 4096. Zero rejects a request that would have to wait for a slot. Omit it for the adapter default.
- **Legacy names and notes** --max-queue
- **Related settings** `server_options.queue_timeout`

### `server_options.queue_timeout`

Limit time spent waiting in the Strata HTTP request queue.

- **JSON paths** `settings.server_options.queue_timeout, profiles.<profile>.settings.server_options.queue_timeout`
- **Command line** `--set 'server_options.queue_timeout=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__QUEUE_TIMEOUT`
- **Valid values** Finite seconds from 0.1 through 600. Requires max_queue to be set. Omit it to leave the deadline disabled.
- **Legacy names and notes** --queue-timeout
- **Related settings** `server_options.max_queue`

### `server_options.api_key`

Require a bearer or x-api-key credential for the Strata HTTP API.

- **JSON paths** `settings.server_options.api_key, profiles.<profile>.settings.server_options.api_key`
- **Command line** `--set 'server_options.api_key=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__API_KEY`
- **Valid values** One nonempty secret, or multiple comma-separated secrets. Use a private configuration source.
- **Legacy names and notes** --api-key
- **Related settings** `env.strata_api_key, host`

### `server_options.api_monitor`

Enable Strata's in-memory HTTP request monitor.

- **JSON paths** `settings.server_options.api_monitor, profiles.<profile>.settings.server_options.api_monitor`
- **Command line** `--set 'server_options.api_monitor=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__API_MONITOR`
- **Valid values** Boolean flag, off by default. It retains recent request prompts and replies.
- **Legacy names and notes** --api-monitor
- **Related settings** `server_options.api_key`

### `server_options.batch_groups`

Choose the number of Strata pipeline batch groups.

- **JSON paths** `settings.server_options.batch_groups, profiles.<profile>.settings.server_options.batch_groups`
- **Command line** `--set 'server_options.batch_groups=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__BATCH_GROUPS`
- **Valid values** auto or a positive integer.
- **Legacy names and notes** --batch-groups
- **Related settings** `server_options.batch_telemetry`

### `server_options.batch_telemetry`

Enable low-rate Strata decode-window telemetry.

- **JSON paths** `settings.server_options.batch_telemetry, profiles.<profile>.settings.server_options.batch_telemetry`
- **Command line** `--set 'server_options.batch_telemetry=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__BATCH_TELEMETRY`
- **Valid values** Boolean flag. Opt in only when collecting batch telemetry.
- **Legacy names and notes** --batch-telemetry
- **Related settings** `server_options.no_batch_telemetry`

### `server_options.before_load`

Run a command before loading the Strata worker.

- **JSON paths** `settings.server_options.before_load, profiles.<profile>.settings.server_options.before_load`
- **Command line** `--set 'server_options.before_load=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__BEFORE_LOAD`
- **Valid values** Command string. Supply a string or an argument array through the run file.
- **Legacy names and notes** --before-load
- **Related settings** `server_options.offline, server_options.lazy`

### `server_options.fit_max_tokens`

Clamp the requested output budget to the available context.

- **JSON paths** `settings.server_options.fit_max_tokens, profiles.<profile>.settings.server_options.fit_max_tokens`
- **Command line** `--set 'server_options.fit_max_tokens=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__FIT_MAX_TOKENS`
- **Valid values** Boolean flag. The Strata adapter normally enables this behavior.
- **Legacy names and notes** --fit-max-tokens
- **Related settings** `server_options.no_fit_max_tokens`

### `server_options.gguf_py`

Locate the GGUF Python helpers used by the Megla adapter.

- **JSON paths** `settings.server_options.gguf_py, profiles.<profile>.settings.server_options.gguf_py`
- **Command line** `--set 'server_options.gguf_py=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__GGUF_PY`
- **Valid values** Directory containing compatible GGUF Python modules.
- **Legacy names and notes** --gguf-py
- **Related settings** `server_options.tokenizer`

### `server_options.idle_unload`

Unload the Strata worker after inactivity.

- **JSON paths** `settings.server_options.idle_unload, profiles.<profile>.settings.server_options.idle_unload`
- **Command line** `--set 'server_options.idle_unload=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__IDLE_UNLOAD`
- **Valid values** Duration in seconds. Omission uses the run file, and zero disables the timer.
- **Legacy names and notes** --idle-unload
- **Related settings** `server_options.lazy, server_options.min_free_vram_mib`

### `server_options.mcp_config`

Load client-tool definitions into the Strata web application.

- **JSON paths** `settings.server_options.mcp_config, profiles.<profile>.settings.server_options.mcp_config`
- **Command line** `--set 'server_options.mcp_config=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__MCP_CONFIG`
- **Valid values** Path to a JSON file containing an mcpServers object. Strict offline mode excludes MCP subprocesses.
- **Legacy names and notes** --mcp-config
- **Related settings** `server_options.offline`

### `server_options.min_free_vram_mib`

Require free GPU memory before reloading the Strata worker.

- **JSON paths** `settings.server_options.min_free_vram_mib, profiles.<profile>.settings.server_options.min_free_vram_mib`
- **Command line** `--set 'server_options.min_free_vram_mib=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__MIN_FREE_VRAM_MIB`
- **Valid values** Nonnegative integer in MiB. Zero leaves the gate disabled.
- **Legacy names and notes** --min-free-vram-mib
- **Related settings** `server_options.lazy, server_options.idle_unload`

### `server_options.no_api_key`

Select the Strata adapter's explicit unauthenticated mode.

- **JSON paths** `settings.server_options.no_api_key, profiles.<profile>.settings.server_options.no_api_key`
- **Command line** `--set 'server_options.no_api_key=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__NO_API_KEY`
- **Valid values** Boolean flag. Anyone who can reach the listener can use it in this mode.
- **Legacy names and notes** --no-api-key
- **Related settings** `server_options.api_key`

### `server_options.no_batch_telemetry`

Disable Strata batch telemetry.

- **JSON paths** `settings.server_options.no_batch_telemetry, profiles.<profile>.settings.server_options.no_batch_telemetry`
- **Command line** `--set 'server_options.no_batch_telemetry=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__NO_BATCH_TELEMETRY`
- **Valid values** Boolean flag. Use it to override a telemetry-enabled setting.
- **Legacy names and notes** --no-batch-telemetry
- **Related settings** `server_options.batch_telemetry`

### `server_options.no_fit_max_tokens`

Reject requests that exceed the available context instead of clamping them.

- **JSON paths** `settings.server_options.no_fit_max_tokens, profiles.<profile>.settings.server_options.no_fit_max_tokens`
- **Command line** `--set 'server_options.no_fit_max_tokens=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__NO_FIT_MAX_TOKENS`
- **Valid values** Boolean flag. A prompt that alone exceeds context is always rejected.
- **Legacy names and notes** --no-fit-max-tokens
- **Related settings** `server_options.fit_max_tokens`

### `server_options.offline`

Enable Strata's strict offline serving policy.

- **JSON paths** `settings.server_options.offline, profiles.<profile>.settings.server_options.offline`
- **Command line** `--set 'server_options.offline=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__OFFLINE`
- **Valid values** Boolean flag. It restricts external image fetches and disallows tool subprocesses and before-load hooks.
- **Legacy names and notes** --offline
- **Related settings** `server_options.mcp_config, server_options.before_load`

### `server_options.open`

Open the Strata web application when the adapter is ready.

- **JSON paths** `settings.server_options.open, profiles.<profile>.settings.server_options.open`
- **Command line** `--set 'server_options.open=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__OPEN`
- **Valid values** Boolean flag. Omission uses the adapter's run settings.
- **Legacy names and notes** --open
- **Related settings** `server_options.ready_file`

### `server_options.ready_file`

Write the Strata instance record after listener startup.

- **JSON paths** `settings.server_options.ready_file, profiles.<profile>.settings.server_options.ready_file`
- **Command line** `--set 'server_options.ready_file=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__READY_FILE`
- **Valid values** Writable path. The record includes bound port and model state.
- **Legacy names and notes** --ready-file
- **Related settings** `server_options.lazy`

### `server_options.replicas`

Divide available Strata GPUs among multiple workers.

- **JSON paths** `settings.server_options.replicas, profiles.<profile>.settings.server_options.replicas`
- **Command line** `--set 'server_options.replicas=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__REPLICAS`
- **Valid values** Integer count accepted by the adapter. For explicit GPU groups, use the run-file replicas array.
- **Legacy names and notes** --replicas
- **Related settings** `server_options.batch_groups, server_options.parallel`

### `server_options.shutdown_timeout`

Set the Strata graceful-drain deadline.

- **JSON paths** `settings.server_options.shutdown_timeout, profiles.<profile>.settings.server_options.shutdown_timeout`
- **Command line** `--set 'server_options.shutdown_timeout=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__SHUTDOWN_TIMEOUT`
- **Valid values** Finite number of seconds from 0 through 300. Omission uses the adapter's normal shutdown behavior.
- **Legacy names and notes** --shutdown-timeout

### `server_options.slot_save_path`

Enable Strata conversation saves and restores through the slots API.

- **JSON paths** `settings.server_options.slot_save_path, profiles.<profile>.settings.server_options.slot_save_path`
- **Command line** `--set 'server_options.slot_save_path=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__SLOT_SAVE_PATH`
- **Valid values** Directory path for saved slot state.
- **Legacy names and notes** --slot-save-path

### `server_options.split_serve`

Use separate Strata workers for prompt processing and decoding.

- **JSON paths** `settings.server_options.split_serve, profiles.<profile>.settings.server_options.split_serve`
- **Command line** `--set 'server_options.split_serve=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__SPLIT_SERVE`
- **Valid values** Two comma-separated GPU ordinals, PREFILL,DECODE, in an adapter configuration that supports this mode.
- **Legacy names and notes** --split-serve
- **Related settings** `server_options.replicas`

### `server_options.strict_model`

Require the requested model name to match the loaded model or its alias.

- **JSON paths** `settings.server_options.strict_model, profiles.<profile>.settings.server_options.strict_model`
- **Command line** `--set 'server_options.strict_model=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__STRICT_MODEL`
- **Valid values** Boolean flag. Strata run files can also enable strict_model.
- **Legacy names and notes** --strict-model
- **Related settings** `server_options.alias`

### `server_options.tok`

Select the persistent Megla tokenizer helper.

- **JSON paths** `settings.server_options.tok, profiles.<profile>.settings.server_options.tok`
- **Command line** `--set 'server_options.tok=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__TOK`
- **Valid values** Path to the compatible executable. Omit it to use the launcher's packaged choice.
- **Legacy names and notes** --tok
- **Related settings** `server_options.tokenizer`

### `server_options.tokenizer`

Specify the HTTP adapter's tokenizer.

- **JSON paths** `settings.server_options.tokenizer, profiles.<profile>.settings.server_options.tokenizer`
- **Command line** `--set 'server_options.tokenizer=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__TOKENIZER`
- **Valid values** Megla accepts a tokenizer executable. Strata's native adapter flag uses a tokenizer directory; configured Strata serving supplies it through strata_settings.tokenizer.
- **Legacy names and notes** --tokenizer
- **Related settings** `server_options.tok, server_options.gguf_py`

### `server_options.parallel`

Request a Strata HTTP slot count without editing the imported run.

- **JSON paths** `settings.server_options.parallel, profiles.<profile>.settings.server_options.parallel`
- **Command line** `--set 'server_options.parallel=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__PARALLEL`
- **Valid values** Integer from 1 through 1024, subject to the worker limit. An explicit batch or slots flag in the saved worker args conflicts with this option.
- **Legacy names and notes** --parallel when passed to the Strata adapter directly. In configured mode use server_options to identify the adapter.
- **Related settings** `server_options.batch_groups`

### `server_options.gpu`

Select GPU ordinals for the Strata adapter.

- **JSON paths** `settings.server_options.gpu, profiles.<profile>.settings.server_options.gpu`
- **Command line** `--set 'server_options.gpu=JSON'`
- **Environment** `SP100_SERVER_OPTIONS__GPU`
- **Valid values** Comma-separated GPU identifiers understood by the adapter. Use the worker option map for GPU flags with worker-specific syntax.
- **Legacy names and notes** --gpu · --gpu when passed to the Strata adapter directly. In configured mode use server_options to identify the adapter.
- **Related settings** `server_options.replicas`

## Worker and server environment

### `env.<variable>`

Configure any supported worker or adapter environment variable.

- **JSON paths** `settings.env.<variable>, profiles.<profile>.settings.env.<variable>`
- **Command line** `--set 'env.<variable>=JSON'`
- **Environment** `SP100_ENV__<VARIABLE>`
- **Valid values** Use the variable name in lowercase in JSON and uppercase after ENV__. The selected component defines the interpretation and range.
- **Related settings** `env, environment_prefix, legacy_names`

### `env.strata_api_key`

Supply the Strata API credential through the service process environment.

- **JSON paths** `settings.env.strata_api_key, profiles.<profile>.settings.env.strata_api_key`
- **Command line** `--set 'env.strata_api_key=JSON'`
- **Environment** `SP100_ENV__STRATA_API_KEY`
- **Valid values** Nonempty secret. An existing service-manager credential remains recognized.
- **Legacy names and notes** STRATA_API_KEY
- **Related settings** `server_options.api_key`

### `env.cuda_visible_devices`

Limit and order CUDA devices visible to the text worker.

- **JSON paths** `settings.env.cuda_visible_devices, profiles.<profile>.settings.env.cuda_visible_devices`
- **Command line** `--set 'env.cuda_visible_devices=JSON'`
- **Environment** `SP100_ENV__CUDA_VISIBLE_DEVICES`
- **Valid values** CUDA visibility list of ordinals or UUIDs, or empty text to hide all devices. Separate projector placement has its own visibility.
- **Legacy names and notes** CUDA_VISIBLE_DEVICES
- **Related settings** `engine_options.gpus, vision_device`

### `env.ink2_ckpt_mb`

Set Megla's checkpoint memory budget.

- **JSON paths** `settings.env.ink2_ckpt_mb, profiles.<profile>.settings.env.ink2_ckpt_mb`
- **Command line** `--set 'env.ink2_ckpt_mb=JSON'`
- **Environment** `SP100_ENV__INK2_CKPT_MB`
- **Valid values** Integer MiB. Megla's checkpoint engine controls its upper bound and eviction policy.
- **Legacy names and notes** INK2_CKPT_MB
- **Related settings** `env.ink2_park_mb, context`

### `env.ink2_park_mb`

Set Megla's parked-conversation memory budget.

- **JSON paths** `settings.env.ink2_park_mb, profiles.<profile>.settings.env.ink2_park_mb`
- **Command line** `--set 'env.ink2_park_mb=JSON'`
- **Environment** `SP100_ENV__INK2_PARK_MB`
- **Valid values** Integer MiB. The worker uses this budget when parking conversation state.
- **Legacy names and notes** INK2_PARK_MB
- **Related settings** `env.ink2_ckpt_mb`

### `env.ink2_vision_timeout_seconds`

Set the Megla per-image deadline across queueing, initialization, and encoding.

- **JSON paths** `settings.env.ink2_vision_timeout_seconds, profiles.<profile>.settings.env.ink2_vision_timeout_seconds`
- **Command line** `--set 'env.ink2_vision_timeout_seconds=JSON'`
- **Environment** `SP100_ENV__INK2_VISION_TIMEOUT_SECONDS`
- **Valid values** Nonnegative seconds. Default 0 leaves it disabled.
- **Legacy names and notes** INK2_VISION_TIMEOUT_SECONDS
- **Related settings** `vision_device, server_options.vis`

### `env.ink2_oa_lookup`

Set Megla HTTP's default prompt-lookup window.

- **JSON paths** `settings.env.ink2_oa_lookup, profiles.<profile>.settings.env.ink2_oa_lookup`
- **Command line** `--set 'env.ink2_oa_lookup=JSON'`
- **Environment** `SP100_ENV__INK2_OA_LOOKUP`
- **Valid values** Integer accepted by Megla. Existing adapter default is 15. A request can supply lookup.
- **Legacy names and notes** INK2_OA_LOOKUP
- **Related settings** `engine_options.lookup`

### `env.ink2_pictures`

Control automatic Megla projector setup when conventional files are available.

- **JSON paths** `settings.env.ink2_pictures, profiles.<profile>.settings.env.ink2_pictures`
- **Command line** `--set 'env.ink2_pictures=JSON'`
- **Environment** `SP100_ENV__INK2_PICTURES`
- **Valid values** Text 0 disables automatic setup. Normal default enables it.
- **Legacy names and notes** INK2_PICTURES
- **Related settings** `vision_device, data_dir`

## Strata run configuration

### `strata_settings.<field>`

Configure a field belonging to the Strata run format.

- **JSON paths** `settings.strata_settings.<field>, profiles.<profile>.settings.strata_settings.<field>`
- **Command line** `--set 'strata_settings.<field>=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Use the field as defined by the pinned Strata adapter. For nested fields on the command line, pass the JSON object for the first key. Environment inputs use the whole STRATA_SETTINGS object.
- **Related settings** `strata_settings, strata_config`

### `strata_settings.lazy_load`

Defer Strata worker loading in the run configuration.

- **JSON paths** `settings.strata_settings.lazy_load, profiles.<profile>.settings.strata_settings.lazy_load`
- **Command line** `--set 'strata_settings.lazy_load=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Boolean. true activates lazy loading. An explicit server lazy flag also enables it.
- **Related settings** `server_options.lazy`

### `strata_settings.tokenizer`

Locate the tokenizer files for Strata serving.

- **JSON paths** `settings.strata_settings.tokenizer, profiles.<profile>.settings.strata_settings.tokenizer`
- **Command line** `--set 'strata_settings.tokenizer=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Nonempty path to supported tokenizer assets. Required by configured Strata HTTP unless supplied by the imported run.
- **Related settings** `strata_config, model`

### `strata_settings.cwd`

Set the Strata worker working directory.

- **JSON paths** `settings.strata_settings.cwd, profiles.<profile>.settings.strata_settings.cwd`
- **Command line** `--set 'strata_settings.cwd=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Nonempty directory path. Defaults to the launch directory. Relative paths in saved worker arguments use this directory.
- **Related settings** `strata_config, model`

### `strata_settings.drafting`

Choose the Strata run's drafting policy.

- **JSON paths** `settings.strata_settings.drafting, profiles.<profile>.settings.strata_settings.drafting`
- **Command line** `--set 'strata_settings.drafting=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** The layer adapter accepts auto, mtp, dflash2, or off where supported. The split planner honors off in the absence of an explicit spec flag.
- **Related settings** `engine_options.draft, engine_options.mtp, engine_options.spec`

### `strata_settings.user_args`

Supply Strata user worker arguments in its saved-run format.

- **JSON paths** `settings.strata_settings.user_args, profiles.<profile>.settings.strata_settings.user_args`
- **Command line** `--set 'strata_settings.user_args=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Array of strings. Uses native flag and operand spelling. Higher-priority arrays replace the entire list.
- **Related settings** `engine_options, strata_config`

### `strata_settings.vision`

Supply the Strata projector configuration.

- **JSON paths** `settings.strata_settings.vision, profiles.<profile>.settings.strata_settings.vision`
- **Command line** `--set 'strata_settings.vision=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** JSON object. External Strata Inkling vision requires mmproj and model. CUDA placement additionally requires the staged adapter and a capable reader.
- **Related settings** `strata_settings.vision.mmproj, strata_settings.vision.model, strata_settings.vision.exe, vision_device`

### `strata_settings.vision.mmproj`

Set the GGUF projector path for Strata's external image reader.

- **JSON paths** `settings.strata_settings.vision.mmproj, profiles.<profile>.settings.strata_settings.vision.mmproj`
- **Command line** `--set 'strata_settings.vision=OBJECT'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Readable model file path.
- **Related settings** `strata_settings.vision, strata_settings.vision.model`

### `strata_settings.vision.model`

Set the text vocabulary GGUF path for the external Strata Inkling projector.

- **JSON paths** `settings.strata_settings.vision.model, profiles.<profile>.settings.strata_settings.vision.model`
- **Command line** `--set 'strata_settings.vision=OBJECT'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Readable compatible GGUF path.
- **Related settings** `strata_settings.vision.mmproj`

### `strata_settings.vision.exe`

Override the external vision helper executable.

- **JSON paths** `settings.strata_settings.vision.exe, profiles.<profile>.settings.strata_settings.vision.exe`
- **Command line** `--set 'strata_settings.vision=OBJECT'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Executable path for the chosen CPU or CUDA projector device.
- **Related settings** `strata_settings.vision, vision_device`

### `strata_settings.split_fallback`

Define the layer worker used if split startup fails.

- **JSON paths** `settings.strata_settings.split_fallback, profiles.<profile>.settings.strata_settings.split_fallback`
- **Command line** `--set 'strata_settings.split_fallback=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Strata run object specifying a fallback executable, arguments, and optional environment and working directory. Configured unified mode points its executable to the bundled layer worker.
- **Related settings** `strata_config, server_options.lazy`

### `strata_settings.backend`

Select the Strata adapter's compute platform.

- **JSON paths** `settings.strata_settings.backend, profiles.<profile>.settings.strata_settings.backend`
- **Command line** `--set 'strata_settings.backend=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Backend accepted by the pinned Strata adapter, such as cuda, cpu, hip, sycl, or metal where available. Launcher backend selects the worker implementation.
- **Related settings** `backend, vision_device`

### `strata_settings.gpu`

Select Strata's GPUs using the run-file interface.

- **JSON paths** `settings.strata_settings.gpu, profiles.<profile>.settings.strata_settings.gpu`
- **Command line** `--set 'strata_settings.gpu=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Strata GPU selection, typically one or more device numbers. Hardware roles and the chosen worker determine the effective visibility.
- **Related settings** `server_options.gpu, engine_options.gpus`

### `strata_settings.layer_split`

Set the Strata run's layer split policy.

- **JSON paths** `settings.strata_settings.layer_split, profiles.<profile>.settings.strata_settings.layer_split`
- **Command line** `--set 'strata_settings.layer_split=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** auto or a layout accepted by the selected Strata worker and adapter. It belongs to the layer path.
- **Related settings** `engine_options.layer_split, strata_settings.gpu`

### `strata_settings.model_name`

Set the model identifier advertised by Strata.

- **JSON paths** `settings.strata_settings.model_name, profiles.<profile>.settings.strata_settings.model_name`
- **Command line** `--set 'strata_settings.model_name=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Model name string used by the adapter and its compatibility routes.
- **Related settings** `model, strata_settings.aliases`

### `strata_settings.aliases`

Add model identifiers that may be used in requests.

- **JSON paths** `settings.strata_settings.aliases, profiles.<profile>.settings.strata_settings.aliases`
- **Command line** `--set 'strata_settings.aliases=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Names in the format accepted by the Strata run file. The adapter checks them when strict model matching is enabled.
- **Related settings** `strata_settings.model_name, server_options.strict_model`

### `strata_settings.max_queue`

Set Strata's pending request capacity in the run file.

- **JSON paths** `settings.strata_settings.max_queue, profiles.<profile>.settings.strata_settings.max_queue`
- **Command line** `--set 'strata_settings.max_queue=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Integer 0–4096. Zero disallows queueing. The server option can override it.
- **Related settings** `server_options.max_queue, strata_settings.queue_timeout`

### `strata_settings.queue_timeout`

Set the Strata run's pending request deadline.

- **JSON paths** `settings.strata_settings.queue_timeout, profiles.<profile>.settings.strata_settings.queue_timeout`
- **Command line** `--set 'strata_settings.queue_timeout=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Finite seconds from 0.1 through 600. Requires max_queue. The server option can override it.
- **Related settings** `strata_settings.max_queue, server_options.queue_timeout`

### `strata_settings.reasoning_budget_tokens`

Set the Strata adapter's default reasoning token budget.

- **JSON paths** `settings.strata_settings.reasoning_budget_tokens, profiles.<profile>.settings.strata_settings.reasoning_budget_tokens`
- **Command line** `--set 'strata_settings.reasoning_budget_tokens=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Nonnegative integer. Zero or an omitted value leaves the configurable cap unset.
- **Related settings** `sampling, tool_streaming`

### `strata_settings.fit_max_tokens`

Choose whether to clamp the output budget to the remaining context.

- **JSON paths** `settings.strata_settings.fit_max_tokens, profiles.<profile>.settings.strata_settings.fit_max_tokens`
- **Command line** `--set 'strata_settings.fit_max_tokens=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Boolean. The pinned adapter defaults to clamping.
- **Related settings** `context, server_options.no_fit_max_tokens`

### `strata_settings.anthropic_thinking`

Control Strata thinking for Anthropic requests that omit an explicit choice.

- **JSON paths** `settings.strata_settings.anthropic_thinking, profiles.<profile>.settings.strata_settings.anthropic_thinking`
- **Command line** `--set 'strata_settings.anthropic_thinking=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** model or on_request.
- **Related settings** `strata_settings.reasoning_budget_tokens`

### `strata_settings.effort_position`

Choose where the Strata adapter places reasoning effort in the prompt.

- **JSON paths** `settings.strata_settings.effort_position, profiles.<profile>.settings.strata_settings.effort_position`
- **Command line** `--set 'strata_settings.effort_position=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** start (default) or end. The end position can retain more cached prompt when effort changes.
- **Related settings** `strata_settings.reasoning_budget_tokens`

### `strata_settings.idle_unload_s`

Set idle unloading in the Strata run file.

- **JSON paths** `settings.strata_settings.idle_unload_s, profiles.<profile>.settings.strata_settings.idle_unload_s`
- **Command line** `--set 'strata_settings.idle_unload_s=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Nonnegative duration in seconds. Zero or omission leaves idle unloading disabled.
- **Related settings** `strata_settings.lazy_load, server_options.idle_unload`

### `strata_settings.mtp`

Choose whether the Strata adapter enables its configured draft layer.

- **JSON paths** `settings.strata_settings.mtp, profiles.<profile>.settings.strata_settings.mtp`
- **Command line** `--set 'strata_settings.mtp=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Boolean. false requests startup without MTP. The split planner honors it when spec is not set explicitly.
- **Related settings** `strata_settings.drafting, engine_options.spec`

### `strata_settings.engine_silence_s`

Limit how long the Strata adapter waits for worker output.

- **JSON paths** `settings.strata_settings.engine_silence_s, profiles.<profile>.settings.strata_settings.engine_silence_s`
- **Command line** `--set 'strata_settings.engine_silence_s=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Nonnegative seconds. The adapter default is 300. Zero disables this worker silence deadline.
- **Related settings** `server_options.queue_timeout`

### `strata_settings.api_monitor`

Enable Strata request monitoring through the run file.

- **JSON paths** `settings.strata_settings.api_monitor, profiles.<profile>.settings.strata_settings.api_monitor`
- **Command line** `--set 'strata_settings.api_monitor=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Boolean. The adapter keeps recent request prompts and replies in memory when enabled.
- **Related settings** `server_options.api_monitor`

### `strata_settings.open_browser`

Open the Strata chat page when the model is ready.

- **JSON paths** `settings.strata_settings.open_browser, profiles.<profile>.settings.strata_settings.open_browser`
- **Command line** `--set 'strata_settings.open_browser=JSON'`
- **Environment** `SP100_STRATA_SETTINGS (whole JSON object)`
- **Valid values** Boolean. Strata opens the browser when --open is supplied and this value is absent or true. False vetoes --open.
- **Related settings** `server_options.open`
