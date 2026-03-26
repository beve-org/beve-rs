# CLI Tool

The crate includes `beve-cli`, a command-line tool for converting between
BEVE, JSON, and MATLAB formats.

## Installation

```bash
cargo install beve --bin beve-cli
```

## Usage

```
beve-cli <command> [options] <input> [output]
```

### Commands

| Command | Description |
|---|---|
| `to-json` | Convert a BEVE file to JSON |
| `to-mat` | Convert a BEVE file to MATLAB v7.3 MAT |
| `from-json` | Convert a JSON file to BEVE |

### Examples

```bash
# BEVE -> JSON (output inferred as data.json)
beve-cli to-json data.beve

# BEVE -> MATLAB (output inferred as data.mat)
beve-cli to-mat data.beve

# Explicit output path
beve-cli to-mat data.beve output.mat

# JSON -> BEVE (output inferred as data.beve)
beve-cli from-json data.json
```

### MAT Options

The `to-mat` command supports additional flags:

| Flag | Description | Default |
|---|---|---|
| `--name <var>` | MATLAB variable name | `data` |
| `--workspace` | Expand top-level object keys into separate workspace variables | off |

```bash
# Name the MATLAB variable "samples"
beve-cli to-mat --name samples recording.beve

# Expand object keys into workspace variables
beve-cli to-mat --workspace config.beve
```
