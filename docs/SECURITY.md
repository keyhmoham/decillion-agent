# Security notes

* **Docker socket.** The backend creates sandbox containers through the host Docker daemon
  (`/var/run/docker.sock`). Access to that socket is root-equivalent on the host. For anything
  beyond a single-user machine, put the daemon behind a filtering proxy (e.g. `tecnativa/docker-socket-proxy`
  allowing only container/exec/volume operations) or use a rootless/remote Docker host, and set `DOCKER_HOST`.
* **Sandbox containers**: non-root user, `cap_drop: ALL`, `no-new-privileges`, memory/CPU/PID limits,
  on the default bridge network (cannot reach the compose network: Postgres, Ollama, backend).
  Set `SANDBOX_NETWORK=none` for fully offline agents.
* **Web tools run inside the sandbox** and refuse private/link-local/loopback targets (also after redirects).
* **Authentication**: set `APP_API_KEY`; REST requires `x-api-key`/`Authorization: Bearer`, WebSocket and
  file URLs accept `?token=`. There are no user accounts - put the stack behind your SSO/reverse proxy for multi-user use.
* **Workspace files served to the browser** carry `Content-Security-Policy: sandbox` and are never rendered
  as HTML in the app origin; `show_html` cards run in an `allow-scripts`-only, network-less iframe.
* **Human approval**: sessions with *Ask before changes* require approval for `bash`, `write_file`, `edit_file`, `bash_job`.
* The model is *abliterated/uncensored*: treat all agent actions as untrusted, keep sandboxes isolated.
