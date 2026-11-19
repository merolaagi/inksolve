"""Long work (downloads, training, the experiment) runs one job at a time in a background
thread; the page polls for progress. Finished jobs are kept in data/jobs.json."""
import json
import threading
import time
import traceback
import uuid
from pathlib import Path


class Jobs:
    def __init__(self, data_dir: Path):
        self.file = Path(data_dir) / "jobs.json"
        self.lock = threading.Lock()
        self.jobs = {}
        self.queue = []
        try:
            for j in json.loads(self.file.read_text()):
                if j["status"] in ("queued", "running"):
                    j["status"], j["error"] = "error", "InkSolve was restarted while this was running."
                self.jobs[j["id"]] = j
        except Exception:
            pass
        self.cv = threading.Condition(self.lock)
        self.save_lock = threading.Lock()
        self.fns = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind, title, fn):
        jid = uuid.uuid4().hex[:10]
        j = {"id": jid, "kind": kind, "title": title, "status": "queued", "progress": 0.0, "message": "Waiting to start",
             "live": None, "result": None, "error": None, "created": time.time(), "started": None, "finished": None, "stop": False}
        with self.cv:
            self.jobs[jid] = j
            self.fns[jid] = fn
            self.queue.append(jid)
            self.cv.notify()
        self._save()
        return self.public(j)

    def stop(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return None
            if j["status"] == "queued":
                j["status"], j["finished"] = "stopped", time.time()
                self.queue.remove(jid)
            j["stop"] = True
        self._save()
        return self.public(j)

    def get(self, jid):
        j = self.jobs.get(jid)
        return self.public(j) if j else None

    def list(self, limit=30):
        js = sorted(self.jobs.values(), key=lambda j: -j["created"])[:limit]
        return [self.public(j, brief=True) for j in js]

    def active(self):
        return [self.public(j, brief=True) for j in self.jobs.values() if j["status"] in ("queued", "running")]

    @staticmethod
    def public(j, brief=False):
        out = {k: v for k, v in j.items() if k != "stop"}
        if brief:
            out.pop("live", None)
            r = out.get("result")
            if isinstance(r, dict):
                out["result"] = {k: r[k] for k in ("id", "name") if k in r}
        return out

    def _worker(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                jid = self.queue.pop(0)
                j = self.jobs[jid]
                fn = self.fns.pop(jid)
                j["status"], j["started"] = "running", time.time()
            self._save()

            def progress(f, msg, live=None, j=j):
                j["progress"], j["message"] = float(max(0, min(1, f))), msg
                if live is not None:
                    j["live"] = live

            try:
                j["result"] = fn(progress, lambda j=j: j["stop"])
                j["status"], j["progress"], j["message"] = "done", 1.0, "Finished"
            except InterruptedError:
                j["status"], j["message"] = "stopped", "Stopped"
            except Exception as e:  # report, keep the worker alive
                j["status"], j["error"] = "error", str(e) or e.__class__.__name__
                j["trace"] = traceback.format_exc()[-2000:]
            j["finished"] = time.time()
            self._save()

    def _save(self):
        try:
            with self.lock:
                keep = sorted(self.jobs.values(), key=lambda j: -j["created"])[:60]
                data = [{k: v for k, v in j.items() if k not in ("live", "stop")} for j in keep]
            with self.save_lock:                                   # one writer at a time, and never a half-written file
                tmp = self.file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, default=str))
                tmp.replace(self.file)
        except Exception:
            pass
