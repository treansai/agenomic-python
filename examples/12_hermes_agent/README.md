# 12 - Hermes Agent under Agenomic control

Files:

- `hermes-config.yaml`: `$HERMES_HOME/config.yaml` routing Hermes through the
  Agenomic Model Gateway, enabling the `agenomic` plugin, staging skill writes
  and installing the `agenomic-hermes-guard` shell hook.
- `agenomic-adapter.yaml`: optional adapter config file (`AGENOMIC_HERMES_CONFIG`).
- `render_config.py`: offline; renders the configuration and validates both files.

Run a controlled Hermes (see `docs/hermes.md` for install and limits):

```bash
export HERMES_HOME=/srv/hermes
cp hermes-config.yaml "$HERMES_HOME/config.yaml"
export AGENOMIC_HERMES_ENDPOINT=https://agenomic.example
export AGENOMIC_HERMES_RUNTIME_TOKEN=agmhr_...      # passed to Hermes
export AGENOMIC_HERMES_SUPERVISOR_TOKEN=agmhs_...   # kept by the supervisor
agenomic-hermes-supervisor --skills-dir /srv/hermes-skills -- hermes chat
```
