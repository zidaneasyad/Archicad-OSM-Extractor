"""Thin HTTP client for Archicad's JSON server (native API + Tapir add-on)."""
from __future__ import annotations

import json
import time

import requests

DEFAULT_URL = "http://127.0.0.1:19723"


class ArchicadError(RuntimeError):
    def __init__(self, message, code=None, command=None):
        super().__init__(message)
        self.code = code
        self.command = command


class Client:
    def __init__(self, url: str | None = None, timeout: float = 120.0, retries: int = 2):
        self.url = url or DEFAULT_URL   # DEFAULT_URL may be re-pointed at runtime (port discovery)
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._session.post(self.url, data=data, timeout=self.timeout)
                resp.raise_for_status()
                return json.loads(resp.content.decode("utf-8"))
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise ArchicadError(f"Archicad not reachable at {self.url}: {last}")

    def call(self, command: str, parameters: dict | None = None) -> dict:
        """Native API call. Returns the 'result' dict; raises ArchicadError on failure."""
        payload = {"command": command}
        if parameters is not None:
            payload["parameters"] = parameters
        resp = self._post(payload)
        if not resp.get("succeeded"):
            err = resp.get("error", {})
            raise ArchicadError(f"{command}: {err.get('code')} {err.get('message')}",
                                err.get("code"), command)
        return resp.get("result", {})

    def tapir(self, command: str, parameters: dict | None = None) -> dict:
        """Tapir add-on command. Returns addOnCommandResponse."""
        res = self.call("API.ExecuteAddOnCommand", {
            "addOnCommandId": {"commandNamespace": "TapirCommand", "commandName": command},
            "addOnCommandParameters": parameters or {},
        })
        resp = res.get("addOnCommandResponse", {})
        if isinstance(resp, dict) and "error" in resp and len(resp) <= 2:
            e = resp["error"]
            raise ArchicadError(f"Tapir {command}: {e.get('code')} {e.get('message')}",
                                e.get("code"), command)
        return resp


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def eid(g):
    """Element id payload {"elementId": {"guid": g}}."""
    return {"elementId": {"guid": g}}


PORT_RANGE = range(19723, 19744)


def probe(port: int, timeout: float = 1.5) -> dict | None:
    """Ask one port for the Archicad product info; None when nothing (Archicad) answers."""
    import socket
    url = f"http://127.0.0.1:{port}"
    try:   # cheap TCP check first: a closed localhost port can take seconds to refuse on Windows
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            pass
    except OSError:
        return None
    try:
        with Client(url, timeout=timeout, retries=0) as c:
            info = c.call("API.GetProductInfo")
        return {"port": port, "url": url, "version": info.get("version"), "buildNumber": info.get("buildNumber"),
                "languageCode": info.get("languageCode")}
    except Exception:  # noqa: BLE001
        return None


def discover(ports=PORT_RANGE) -> dict:
    """Scan the Archicad JSON ports; check Tapir (GetAddOnVersion) on each live one.
    -> {"archicad": [..], "selected": {...}|None, "tapir": {"present", "version", "error"}|None}"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        found = [r for r in ex.map(probe, list(ports)) if r]
    sel = None
    for r in found:
        try:
            with Client(r["url"], timeout=5.0, retries=0) as c:
                v = c.tapir("GetAddOnVersion")
            r["tapir"] = {"present": True, "version": v.get("version")}
        except Exception as ex:  # noqa: BLE001
            r["tapir"] = {"present": False, "error": str(ex)[:200]}
        if sel is None and r["tapir"]["present"]:
            sel = r
    if sel is None and found:
        sel = found[0]
    tapir = sel["tapir"] if sel else None
    return {"archicad": found, "selected": sel, "tapir": tapir}


def use(url: str) -> None:
    """Point every default Client at this Archicad instance."""
    global DEFAULT_URL
    DEFAULT_URL = url
