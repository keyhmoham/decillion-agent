"""DockerSandbox logic against a faithful fake of the docker SDK surface it uses.

(No Docker daemon is needed; the real-daemon path is covered by the first-deployment checklist.)
"""

import io
import tarfile

import pytest

from app.config import Settings
from app.sandbox.docker_sandbox import DockerSandbox


class NotFound(Exception):
    pass


class FakeContainer:
    def __init__(self, name, files):
        self.name, self.id, self.short_id, self.status, self.files = name, "cid123", "cid123"[:6], "running", files
        self.exec_runs = []

    def reload(self): ...
    def start(self): self.status = "running"
    def stop(self, timeout=10): self.status = "exited"
    def remove(self, force=False, v=False): self.status = "removed"

    def exec_run(self, cmd, **kw):
        self.exec_runs.append(cmd)
        return (0, b"")

    def put_archive(self, path, data):
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for m in tar.getmembers():
                self.files[f"{path.rstrip('/')}/{m.name}"] = tar.extractfile(m).read()
        return True

    def get_archive(self, path):
        if path not in self.files:
            raise NotFound(path)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.rsplit("/", 1)[-1])
            info.size = len(self.files[path])
            tar.addfile(info, io.BytesIO(self.files[path]))
        return iter([buf.getvalue()]), {}


class FakeApi:
    def __init__(self, scripted):
        self.scripted, self.created = scripted, []

    def exec_create(self, cid, argv, **kw):
        self.created.append((argv, kw))
        return {"Id": f"ex{len(self.created)}"}

    def exec_start(self, eid, stream=False, demux=False):
        out = self.scripted(self.created[-1][0], self.created[-1][1])
        yield from out

    def exec_inspect(self, eid):
        return {"ExitCode": self.created and getattr(self, "code", 0)}


class FakeContainers:
    def __init__(self, store):
        self.store, self.run_kwargs = store, None

    def get(self, name):
        if name not in self.store:
            raise NotFound(name)
        return self.store[name]

    def run(self, image, **kw):
        self.run_kwargs = {"image": image, **kw}
        c = FakeContainer(kw["name"], {})
        self.store[kw["name"]] = c
        return c

    def list(self, **kw):
        return list(self.store.values())


class FakeClient:
    def __init__(self, scripted=lambda argv, kw: iter([(b"", None)])):
        self.store = {}
        self.containers = FakeContainers(self.store)
        self.api = FakeApi(scripted)

        class Vols:
            def get(self_, n):
                return type("V", (), {"remove": lambda s, force=False: None})()

        self.volumes = Vols()


@pytest.fixture(autouse=True)
def patch_docker_errors(monkeypatch):
    import docker.errors as de

    monkeypatch.setattr(de, "NotFound", NotFound)
    monkeypatch.setattr(de, "ImageNotFound", type("ImageNotFound", (Exception,), {}))
    monkeypatch.setattr(de, "APIError", type("APIError", (Exception,), {"explanation": ""}))


def box(client, **kw):
    s = Settings(sandbox_backend="docker", sandbox_image="img:1", sandbox_memory="1g", sandbox_cpus=1.5, **kw)
    return DockerSandbox("0123abcd-0000-0000-0000-000000000000", s, client=client), s


async def test_container_is_created_hardened_with_workspace_volume():
    client = FakeClient()
    sb, s = box(client)
    await sb.ensure()
    kw = client.containers.run_kwargs
    assert kw["image"] == "img:1" and kw["name"] == "a1-sbx-0123abcd0000"
    assert kw["cap_drop"] == ["ALL"] and "no-new-privileges:true" in kw["security_opt"] and kw["init"] is True
    assert kw["user"] == "agent" and kw["mem_limit"] == "1g" and kw["nano_cpus"] == int(1.5e9) and kw["pids_limit"] == 512
    assert kw["volumes"] == {"a1-ws-0123abcd0000": {"bind": "/workspace", "mode": "rw"}}
    assert kw["labels"]["decillion.sandbox"] == "1" and kw["labels"]["decillion.session"].startswith("0123abcd")
    await sb.ensure()  # idempotent: no second creation
    assert len(client.store) == 1


async def test_exec_builds_safe_command_and_collects_streams():
    seen = {}

    def script(argv, kw):
        seen.update(argv=argv, kw=kw)
        return iter([(b"hello\n", None), (None, b"warn\n"), (b"bye\n", None)])

    client = FakeClient(script)
    client.api.code = 0
    sb, _ = box(client)
    chunks = []
    res = await sb.exec("echo 'it''s' && ls | wc -l", timeout=30, on_output=lambda s, t: chunks.append((s, t)))
    assert res.stdout == "hello\nbye\n" and res.stderr == "warn\n" and res.ok
    # the user command travels in the environment, never through argv quoting
    assert seen["kw"]["environment"]["A1_CMD"] == "echo 'it''s' && ls | wc -l"
    assert seen["argv"][:5] == ["setsid", "-w", "timeout", "-k", "3"] and seen["argv"][5] == "30"
    assert seen["argv"][-2:] == ['eval "$A1_CMD"', seen["argv"][-1]] or seen["argv"][-2] == 'eval "$A1_CMD"'
    assert seen["kw"]["user"] == "agent" and seen["kw"]["workdir"] == "/workspace"
    assert ("stdout", "hello\n") in chunks and ("stderr", "warn\n") in chunks


async def test_exit_codes_and_timeouts():
    client = FakeClient(lambda a, k: iter([(b"", None)]))
    sb, _ = box(client)
    client.api.code = 3
    assert (await sb.exec("false")).exit_code == 3
    client.api.code = 124
    res = await sb.exec("sleep 9", timeout=1)
    assert res.exit_code == 124  # fast fake => not flagged as a real timeout
    assert (await sb.exec("x", timeout=99999)) is not None
    assert client.api.created[-1][0][5] == str(sb.s.sandbox_max_command_timeout)  # clamped


async def test_file_roundtrip_through_tar_archives():
    client = FakeClient()
    sb, _ = box(client)
    await sb.write_bytes("/workspace/src/a.txt", b"hello \xc3\xa9")
    c = client.store["a1-sbx-0123abcd0000"]
    assert c.files["/workspace/src/a.txt"] == b"hello \xc3\xa9"
    assert await sb.read_bytes("/workspace/src/a.txt") == b"hello \xc3\xa9"
    with pytest.raises(FileNotFoundError):
        await sb.read_bytes("/workspace/missing")
    assert sb.resolve("src/a.txt") == "/workspace/src/a.txt" and sb.resolve("../etc") == "/etc"


async def test_list_dir_parsing_and_exists():
    listing = b"d\t4096\t1700000000.5\tsrc\nf\t12\t1700000001.0\tREADME.md\nf\t5\t1700000002.0\t.env\n"

    def script(argv, kw):
        return iter([(listing if "find" in kw["environment"]["A1_CMD"] else b"dir\n", None)])

    client = FakeClient(script)
    sb, _ = box(client)
    entries = await sb.list_dir("/workspace")
    assert [e.name for e in entries] == ["src", ".env", "README.md"]  # dirs first, then alphabetical
    assert entries[0].is_dir and entries[2].size == 12
    assert await sb.exists("/workspace") == "dir"


async def test_mcp_spec_runs_inside_the_sandbox_via_sdk_not_the_docker_cli():
    sb, _ = box(FakeClient())
    spec = sb.mcp_spec(["mcp-server-time"], {"TOKEN": "x"})
    assert spec == {"transport": "sandbox_exec", "argv": ["mcp-server-time"], "env": {"TOKEN": "x"}}
    assert "command" not in spec  # nothing is spawned on the backend host


async def test_stop_and_destroy():
    client = FakeClient()
    sb, _ = box(client)
    await sb.ensure()
    await sb.stop()
    assert client.store["a1-sbx-0123abcd0000"].status == "exited"
    await sb.destroy()
    assert client.store["a1-sbx-0123abcd0000"].status == "removed"
