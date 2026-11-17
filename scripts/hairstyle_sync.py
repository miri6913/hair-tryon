"""Drive the hairstyle model comparison on the GPU server from this PC. One explicit action per call; nothing runs unasked.
Run it from cmd through the wrapper:  scripts\\hs <action> [options]
Needs two environment variables: VTO_HOST (ssh host or alias of the GPU server) and VTO_SRV_ROOT (project folder on it).

  scripts\\hs status                              read-only: GPUs, disk, containers, server folder
  scripts\\hs push                                copy server/ -> code/, data/hairstyle_set -> inputs/
  scripts\\hs build -m qwen_text                  docker build on the server (no GPU)
  scripts\\hs align                               FFHQ-align every photo + framing check (CPU, hairfast image)
  scripts\\hs fetch-weights -m stable_hair       Stable-Hair + SD 1.5 weights (no GPU); also -m hairfast
  scripts\\hs run -m qwen_text -g 1 -l 2          smoke test of 2 jobs; detached container
  scripts\\hs finish -m hairfast -g 0              raw GAN crops -> full-size photos (clean plate + composite)
  scripts\\hs logs -m qwen_text
  scripts\\hs pull -m qwen_text                   outputs -> results/hairstyle/<model>
  scripts\\hs clean -m qwen_text                  remove the finished container
  scripts\\hs api-build | api-start -g N | api-stop | api-logs   the hairstyle API container (server/hairapi)
  scripts\\hs api-test-start -g N --refs-dir refs_nano | api-test-stop | api-test-logs   a 2nd API on :8091 for candidate references
  scripts\\hs tunnel-start | tunnel-stop | tunnel-logs           the Cloudflare Tunnel container (public HTTPS)
  scripts\\hs tunnel-quick                                       no account/domain: random trycloudflare.com URL, changes on restart
  scripts\\hs funnel-start | funnel-stop | funnel-logs           Tailscale Funnel container: free, permanent https://vto-hair.<tailnet>.ts.net

`run` needs an explicit -g and refuses if that GPU already holds more than --max-busy MiB: the server is shared.
Everything on the server stays under SRV (Docker image layers aside). The PC copy of the code is the source of truth.
Needs only ssh and scp (built into Windows) and the standard library.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = os.environ.get("VTO_HOST") or sys.exit("set VTO_HOST to the ssh host (name or alias) of your GPU server")
SRV = os.environ.get("VTO_SRV_ROOT") or sys.exit("set VTO_SRV_ROOT to the project folder on that server (it holds code/ inputs/ outputs/ models/ hf-cache/)")
FLUX2 = ["flux2_4b_text", "flux2_4b_ref", "flux2_9b_text", "flux2_9b_ref"]
MODELS = ["qwen_text", "qwen_ref", "stable_hair", "hairfast", "hairclipv2"] + FLUX2 + ["flux2_refs"]   # flux2_refs: synthetic catalogue references
GAN_MODELS = ["hairfast", "hairclipv2"]   # outputs/<model>/{raw,final,plates}; the others write flat folders
TAILSCALE_TAG = "v1.80.3"   # pinned; bump deliberately
SSH = ["ssh", "-o", "BatchMode=yes", HOST]
SCP = ["scp", "-q", "-o", "BatchMode=yes"]


def remote(cmd: str, capture: bool = False) -> str:
    r = subprocess.run(SSH + [cmd], text=True, capture_output=capture)
    if r.returncode != 0:
        sys.exit(f"remote command failed ({r.returncode}): {cmd}\n{r.stderr if capture else ''}")
    return r.stdout if capture else ""


def scp(args: list[str], cwd: Path) -> None:
    # relative paths + cwd: scp reads "E:\..." as host "E"
    r = subprocess.run(SCP + args, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"scp failed: {args}")


def push_dir(local: Path, remote_dir: str, skip: tuple[str, ...] = ()) -> None:
    remote(f"mkdir -p {remote_dir}")
    files = [p.name for p in sorted(local.iterdir()) if p.is_file()]
    if files:
        scp(files + [f"{HOST}:{remote_dir}/"], local)
    for d in sorted(local.iterdir()):
        if d.is_dir() and d.name != "__pycache__" and d.name not in skip:
            push_dir(d, f"{remote_dir}/{d.name}")


def code_dir(model: str) -> str:
    if model in FLUX2 or model == "flux2_refs":
        return "flux2"
    return model if model in ("stable_hair", "hairfast", "hairclipv2") else "qwen"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "push", "build", "align", "fetch-weights", "run", "finish", "logs", "pull", "clean",
                                       "api-build", "api-start", "api-stop", "api-logs", "api-test-start", "api-test-stop", "api-test-logs",
                                       "tunnel-start", "tunnel-quick", "tunnel-stop", "tunnel-logs",
                                       "funnel-start", "funnel-stop", "funnel-logs"])
    ap.add_argument("-m", "--model", choices=MODELS)
    ap.add_argument("-g", "--gpu", type=int, default=-1)
    ap.add_argument("-l", "--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
    ap.add_argument("--max-busy", type=int, default=2000, help="refuse a GPU that already uses more than this many MiB")
    ap.add_argument("--extra", default="", help='run: extra arguments for run.py, written --extra="--job X --sweep 20:768:4"')
    ap.add_argument("--dir", help="pull: outputs subfolder to fetch (default: the model name), e.g. qwen_text_sweep")
    ap.add_argument("--refs-dir", help="api-test-start: server folder with the candidate references (+ optional catalog.json), e.g. refs_nano or refs")
    ap.add_argument("--test-env", action="append", default=[], metavar="KEY=VAL",
                    help="api-test-start: extra environment for the test container, repeatable, e.g. --test-env REF_MODE=text")
    a = ap.parse_args()
    if a.action in ("build", "fetch-weights", "run", "finish", "logs", "pull", "clean") and not a.model:
        sys.exit(f"-m/--model is required for '{a.action}'")

    if a.action == "status":
        remote("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv; "
               "df -h /mnt/data | tail -1; docker ps -a --filter name=vto- --format '{{.Names}}  {{.Status}}'; "
               f"ls -la {SRV} 2>&1; true")   # the folder does not exist before the first push

    elif a.action == "push":
        remote(f"mkdir -p {SRV}/code {SRV}/inputs {SRV}/outputs {SRV}/models {SRV}/hf-cache")
        push_dir(ROOT / "server", f"{SRV}/code")
        push_dir(ROOT / "src" / "vto_spike", f"{SRV}/code/vto_spike")   # the hairapi image bakes in the shared recolour/segment code
        push_dir(ROOT / "models", f"{SRV}/code/models")                  # ... and the MediaPipe .tflite files
        # Optional local-only data (git-ignored, not in the public repo): reference photos (only for REF_MODE=ref) and your own test photos.
        # The default text-mode API needs none of it.
        refs = ROOT / "data" / "hairapi_refs"
        if refs.is_dir():
            push_dir(refs, f"{SRV}/refs", skip=("incoming",))   # the hairstyle API's catalogue references
            # candidate reference sets (incoming/nano, incoming/real, ...) -> refs_<source>, for `api-test-start --refs-dir refs_<source>`
            incoming = refs / "incoming"
            for src in sorted(incoming.iterdir()) if incoming.is_dir() else []:
                if src.is_dir() and any(src.iterdir()):
                    push_dir(src, f"{SRV}/refs_{src.name}")
        if (ROOT / "data" / "hairstyle_set").is_dir():
            push_dir(ROOT / "data" / "hairstyle_set", f"{SRV}/inputs")
            csvs = [c.name for c in (ROOT / "data").glob("hairstyle_set.csv")]
            if csvs:
                scp(csvs + [f"{HOST}:{SRV}/inputs/"], ROOT / "data")
            remote(f"ls {SRV}/inputs | head -5; echo ...; ls {SRV}/inputs/masks | wc -l")

    elif a.action == "build":
        d = code_dir(a.model)
        # context is code/ (not the model folder) so Dockerfiles can COPY the shared common/
        remote(f"cd {SRV}/code && docker build -f {d}/Dockerfile -t vto-hair-{d} .")

    elif a.action == "align":
        # CPU only (dlib): FFHQ-aligned 1024 crops + framing.csv for every photo in the set, using the hairfast image
        remote(f"docker run --rm --user $(id -u):$(id -g) -v {SRV}:/work vto-hair-hairfast python /code/common/align.py")

    elif a.action == "fetch-weights":
        if a.model not in ("stable_hair", "hairfast", "hairclipv2"):
            sys.exit("only stable_hair, hairfast and hairclipv2 have a separate weights step (Qwen downloads into hf-cache on first run)")
        remote(f"docker run --rm -v {SRV}:/work -e HF_HOME=/work/hf-cache vto-hair-{a.model} sh /code/fetch_weights.sh")

    elif a.action in ("run", "finish"):
        if a.gpu < 0:
            sys.exit("-g/--gpu is required (0 or 1): the server is shared, so pick one on purpose")
        used = int(remote(f"nvidia-smi -i {a.gpu} --query-gpu=memory.used --format=csv,noheader,nounits", capture=True).split()[0])
        if used > a.max_busy:
            sys.exit(f"GPU {a.gpu} is busy ({used} MiB used, limit {a.max_busy}); not starting")
        d = code_dir(a.model)
        if a.action == "finish":
            # raw 1024 crops -> full-size photos (common/finish.py); same container name as run, so logs/pull/clean work the same
            # always the hairfast image: it holds the BiSeNet code from the HairFastGAN repo
            remote(f"docker run -d --name vto-{a.model} --gpus device={a.gpu} -v {SRV}:/work "
                   f"vto-hair-hairfast python /code/common/finish.py --model {a.model} {a.extra}")
            print(f"started vto-{a.model} (finish) on GPU {a.gpu}; follow with: scripts\\hs logs -m {a.model}")
            return
        variant = {"qwen_text": "--variant text", "qwen_ref": "--variant ref"}.get(a.model, "")
        script = "run.py"
        if a.model in FLUX2:
            _, size, var = a.model.split("_")
            variant = f"--size {size} --variant {var}"
        elif a.model == "flux2_refs":
            script = "gen_refs.py"
        limit = f"--limit {a.limit}" if a.limit > 0 else ""
        # HF_TOKEN (from the server shell, if set) is for the gated FLUX.2 klein 9B repo
        # --user: files written under outputs/ belong to the server user, not root (root-owned outputs cannot be deleted by us)
        remote(f"docker run -d --name vto-{a.model} --gpus device={a.gpu} --user $(id -u):$(id -g) -v {SRV}:/work "
               f"-e HF_HOME=/work/hf-cache -e HF_TOKEN vto-hair-{d} python /code/{script} {variant} {limit} {a.extra}")
        print(f"started vto-{a.model} on GPU {a.gpu}; follow with: scripts\\hs logs -m {a.model}")

    elif a.action == "logs":
        remote(f"docker logs --tail 30 vto-{a.model} 2>&1; docker ps -a --filter name=vto-{a.model} --format '{{{{.Status}}}}'")

    elif a.action == "pull":
        dest = ROOT / "results" / "hairstyle"
        dest.mkdir(parents=True, exist_ok=True)
        name = a.dir or a.model
        if a.model in GAN_MODELS and not a.dir:
            # raw/ and plates/ stay on the server; the review sheets use final/
            (dest / name).mkdir(exist_ok=True)
            scp(["-r", f"{HOST}:{SRV}/outputs/{name}/final", "."], dest / name)
            scp([f"{HOST}:{SRV}/outputs/{name}/raw/timings.csv", "."], dest / name)
            print(f"pulled to {dest / name}")
            return
        scp(["-r", f"{HOST}:{SRV}/outputs/{name}", "."], dest)
        shutil.rmtree(dest / name / "_tmp", ignore_errors=True)
        print(f"pulled to {dest / name}")

    elif a.action == "api-build":
        remote(f"cd {SRV}/code && docker build -f hairapi/Dockerfile -t vto-hair-api .")

    elif a.action == "api-start":
        if a.gpu < 0:
            sys.exit("-g/--gpu is required (0 or 1): the server is shared, so pick one on purpose")
        # a restart: our own running API holds the GPU, so remove it before the busy check (the check is for other people's jobs)
        remote("docker rm -f vto-hairapi >/dev/null 2>&1; sleep 3; true")
        used = int(remote(f"nvidia-smi -i {a.gpu} --query-gpu=memory.used --format=csv,noheader,nounits", capture=True).split()[0])
        if used > a.max_busy:
            sys.exit(f"GPU {a.gpu} is busy ({used} MiB used, limit {a.max_busy}); the old API container was removed, nothing started")
        env = f"{SRV}/hairapi.env"
        # the API key lives only in this file on the server (created once, mode 600) and is never printed here:
        # read it yourself with  ssh {HOST} cat {env}
        remote(f"test -f {env} || (umask 077; echo API_KEY=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\n') > {env})")
        remote(f"docker run -d --name vto-hairapi --restart unless-stopped --gpus device={a.gpu} --user $(id -u):$(id -g) "
               # only refs/ (the synthetic catalogue references) is visible to the container, not the personal test photos;
               # MAX_QUEUE 12 x ~7 s = 84 s worst-case wait, under Cloudflare's ~100 s request limit
               f"-p 127.0.0.1:8090:8090 --env-file {env} -e MAX_QUEUE=12 -v {SRV}/hf-cache:/hf:ro -v {SRV}/refs:/inputs:ro vto-hair-api")
        print(f"started vto-hairapi on GPU {a.gpu} (127.0.0.1:8090 on the server); the model takes about a minute to load: scripts\\hs api-logs")

    elif a.action == "tunnel-start":
        # Cloudflare Tunnel: cloudflared dials OUT to Cloudflare, so no inbound port and no sudo. The token (a secret) goes into
        # cloudflared.env on the server by hand, never through this script; the public hostname -> http://localhost:8090 is
        # configured in the Cloudflare dashboard. --network host so it reaches the API on the server's 127.0.0.1:8090.
        env = f"{SRV}/cloudflared.env"
        remote(f"test -f {env} || {{ echo 'missing {env}: see server/hairapi/README.md (Public HTTPS)' >&2; exit 1; }}")
        remote("docker rm -f vto-tunnel >/dev/null 2>&1; true")
        remote(f"docker run -d --name vto-tunnel --restart unless-stopped --network host --env-file {env} "
               "cloudflare/cloudflared:latest tunnel --no-autoupdate run")
        print("started vto-tunnel; check it with: scripts\\hs tunnel-logs")

    elif a.action == "tunnel-quick":
        # no Cloudflare account or domain: a quick tunnel gets a random https://<words>.trycloudflare.com that is valid until this
        # container stops (a new start = a new address; no --restart for that reason). For staging only, no uptime promise.
        remote("docker rm -f vto-tunnel-quick >/dev/null 2>&1; true")
        remote("docker run -d --name vto-tunnel-quick --network host cloudflare/cloudflared:latest "
               "tunnel --no-autoupdate --url http://localhost:8090")
        remote("for i in 1 2 3 4 5 6 7 8 9 10; do u=$(docker logs vto-tunnel-quick 2>&1 | grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' | head -1); "
               "[ -n \"$u\" ] && break; sleep 2; done; echo \"public URL: $u\"")

    elif a.action == "tunnel-stop":
        remote("docker rm -f vto-tunnel vto-tunnel-quick")

    elif a.action == "tunnel-logs":
        remote("docker logs --tail 30 vto-tunnel 2>&1; docker ps -a --filter name=vto-tunnel --format '{{.Names}} {{.Status}}'")

    elif a.action == "funnel-start":
        # Tailscale Funnel: free, fixed https://vto-hair.<tailnet>.ts.net, no domain and no sudo (userspace networking, outbound only).
        # The auth key (a secret) goes into tailscale.env on the server by hand, never through this script. The state dir keeps the
        # node identity, so the key is used once and a restart keeps the same URL. serve.json proxies :443 -> the API on 127.0.0.1:8090.
        env, ts = f"{SRV}/tailscale.env", f"{SRV}/tailscale"
        remote(f"test -f {env} || {{ echo 'missing {env}: see server/hairapi/README.md (Public HTTPS)' >&2; exit 1; }}")
        remote(f"mkdir -p {ts}/state")
        serve = ('{"TCP":{"443":{"HTTPS":true}},'
                 '"Web":{"${TS_CERT_DOMAIN}:443":{"Handlers":{"/":{"Proxy":"http://127.0.0.1:8090"}}}},'
                 '"AllowFunnel":{"${TS_CERT_DOMAIN}:443":true}}')
        remote(f"printf '%s' '{serve}' > {ts}/serve.json")
        remote("docker rm -f vto-funnel >/dev/null 2>&1; true")
        remote(f"docker run -d --name vto-funnel --restart unless-stopped --network host --env-file {env} "
               "-e TS_USERSPACE=true -e TS_HOSTNAME=vto-hair -e TS_STATE_DIR=/state -e TS_SERVE_CONFIG=/config/serve.json "
               f"-v {ts}/state:/state -v {ts}/serve.json:/config/serve.json:ro tailscale/tailscale:{TAILSCALE_TAG}")
        remote("for i in 1 2 3 4 5 6 7 8 9 10 11 12; do u=$(docker exec vto-funnel tailscale status --json 2>/dev/null "
               "| grep -o '\"DNSName\": *\"[^\"]*\"' | head -1 | cut -d'\"' -f4); [ -n \"$u\" ] && break; sleep 3; done; "
               "echo \"public URL: https://${u%.}\"")

    elif a.action == "funnel-stop":
        remote("docker rm -f vto-funnel")

    elif a.action == "funnel-logs":
        remote("docker logs --tail 30 vto-funnel 2>&1; docker exec vto-funnel tailscale funnel status 2>&1; "
               "docker ps -a --filter name=vto-funnel --format '{{.Names}} {{.Status}}'")

    elif a.action == "api-stop":
        remote("docker rm -f vto-hairapi")

    elif a.action == "api-logs":
        remote("docker logs --tail 40 vto-hairapi 2>&1; docker ps -a --filter name=vto-hairapi --format '{{.Status}}'")

    elif a.action == "api-test-start":
        # a SECOND API container for trying candidate reference sets with the exact production pipeline, without touching the live one:
        # port 8091, catalogue from <refs-dir>/catalog.json (python scripts/make_test_catalog.py <source>), no restart policy.
        # It holds a GPU while it runs (~10 GB): stop it with api-test-stop right after the test.
        if a.gpu < 0:
            sys.exit("-g/--gpu is required (0 or 1): the server is shared, so pick one on purpose")
        if not a.refs_dir:
            sys.exit("--refs-dir is required, e.g. --refs-dir refs_nano (pushed from data/hairapi_refs/incoming/nano)")
        remote("docker rm -f vto-hairapi-test >/dev/null 2>&1; sleep 2; true")
        used = int(remote(f"nvidia-smi -i {a.gpu} --query-gpu=memory.used --format=csv,noheader,nounits", capture=True).split()[0])
        if used > a.max_busy:
            sys.exit(f"GPU {a.gpu} is busy ({used} MiB used, limit {a.max_busy}); not starting")
        # a catalog.json in the refs folder (make_test_catalog.py) is used if present; otherwise the image's default catalogue
        has_cat = remote(f"test -f {SRV}/{a.refs_dir}/catalog.json && echo yes || echo no", capture=True).strip() == "yes"
        extra_env = " ".join(f"-e {kv}" for kv in a.test_env)
        remote(f"docker run -d --name vto-hairapi-test --gpus device={a.gpu} --user $(id -u):$(id -g) "
               f"-p 127.0.0.1:8091:8090 --env-file {SRV}/hairapi.env -e MAX_QUEUE=4 {'-e CATALOG=/inputs/catalog.json ' if has_cat else ''}{extra_env} "
               f"-v {SRV}/hf-cache:/hf:ro -v {SRV}/{a.refs_dir}:/inputs:ro vto-hair-api")
        print(f"started vto-hairapi-test on GPU {a.gpu} (127.0.0.1:8091 on the server, refs {a.refs_dir}, "
              f"{'its catalog.json' if has_cat else 'default catalogue'}, env {a.test_env or '-'}); about a minute to load: scripts\\hs api-test-logs")

    elif a.action == "api-test-stop":
        remote("docker rm -f vto-hairapi-test")

    elif a.action == "api-test-logs":
        remote("docker logs --tail 20 vto-hairapi-test 2>&1; docker ps -a --filter name=vto-hairapi-test --format '{{.Status}}'")

    elif a.action == "clean":
        # only removes the container of a finished run; refuses while it is still running
        remote(f"docker ps -q --filter name=vto-{a.model} --filter status=running | grep -q . "
               f"&& echo 'still running, not removing' || docker rm vto-{a.model}")


if __name__ == "__main__":
    main()
