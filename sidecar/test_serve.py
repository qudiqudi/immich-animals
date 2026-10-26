"""Behaviour of serve.py against tiny constant-output ONNX models.

Usage: uv run --project sidecar pytest sidecar
"""

import asyncio
import hashlib
import importlib.util
import json
import threading
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnx
import orjson
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

ENTRIES = {"facial-recognition": {"detection": {"options": {"minScore": 0.7}}}}


def _constant_model(path: Path, input_shape: list[int], output: np.ndarray, **meta):
    """A graph that ignores its input and always returns `output`."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["out"], value=numpy_helper.from_array(output)
            )
        ],
        "fake",
        [helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(output.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(model, meta)
    onnx.save(model, path)


def _embedder(model_dir: Path, stem: str, vector: np.ndarray):
    """An embedder that returns `vector` for every crop in the batch."""
    nodes = [
        helper.make_node("Flatten", ["in"], ["flat"], axis=1),
        helper.make_node("ReduceMean", ["flat"], ["mean"], axes=[1]),
        helper.make_node(
            "Constant", [], ["zero"], value=numpy_helper.from_array(np.float32(0))
        ),
        helper.make_node(
            "Constant", [], ["vector"], value=numpy_helper.from_array(vector[None])
        ),
        helper.make_node("Mul", ["mean", "zero"], ["zeros"]),
        helper.make_node("Add", ["zeros", "vector"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fake",
        [helper.make_tensor_value_info("in", TensorProto.FLOAT, ["n", 3, 8, 8])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["n", 512])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, model_dir / f"{stem}.onnx")
    prep = {"mean": [0.5] * 3, "std": [0.5] * 3}
    (model_dir / f"{stem}.json").write_text(json.dumps({"preprocessing": prep}))


def _load_serve(model_dir: Path, monkeypatch, **env):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "serve", Path(__file__).with_name("serve.py")
    )
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    return serve


def _image() -> bytes:
    return cv2.imencode(".png", np.zeros((640, 640, 3), np.uint8))[1].tobytes()


def test_each_species_gets_its_own_threshold_embedder_and_mapping(
    tmp_path, monkeypatch
):
    rows = [
        [10, 10, 100, 100, 0.50, 0],  # dog
        [200, 200, 300, 300, 0.25, 1],  # cat, kept by CAT_MIN_SCORE=0.2
        [300, 10, 400, 100, 0.35, 1],  # second cat, embedded in the same batch
        [400, 400, 500, 500, 0.10, 1],  # cat below its threshold
        [10, 400, 100, 500, 0.90, 2],  # unknown species
    ]
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog', 1: 'cat', 2: 'bird'}",
    )
    rng = np.random.default_rng(0)
    shared, cat = rng.normal(size=(2, 512)).astype(np.float32)
    _embedder(tmp_path, "embedding", shared)
    _embedder(tmp_path, "embedding_cat", cat)
    serve = _load_serve(
        tmp_path,
        monkeypatch,
        KEEP_HUMAN_FACES="false",
        CAT_MIN_SCORE="0.2",
        DOG_MAX_DISTANCE="0.5",  # equals Immich's, so dog vectors pass through
        CAT_MAX_DISTANCE="0.3",
    )

    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )
    faces = {
        round(f["score"], 2): np.array(json.loads(f["embedding"]))
        for f in response.json()["facial-recognition"]
    }

    assert set(faces) == {0.5, 0.25, 0.35}
    assert np.allclose(faces[0.5], shared)
    for score in (0.25, 0.35):
        cosine = faces[score] @ cat / np.linalg.norm(cat)
        assert abs(cosine - np.sqrt(0.5 / 0.7)) < 0.1  # cat's own shift was applied


def _slow_detector_serve(tmp_path, monkeypatch, **env):
    """serve with a detector that blocks until released; returns (serve, started, release, done)."""
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.zeros((1, 1, 6), np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    serve = _load_serve(tmp_path, monkeypatch, **env)
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def detect(rgb):
        started.set()
        # Bounded, so a blocked event loop fails the test instead of hanging it.
        release.wait(timeout=5)
        done.set()
        return []

    monkeypatch.setattr(serve, "_detect", detect)
    return serve, started, release, done


async def _predict(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )


def test_server_answers_while_animals_are_inferred(tmp_path, monkeypatch):
    serve, started, release, done = _slow_detector_serve(
        tmp_path, monkeypatch, KEEP_HUMAN_FACES="false"
    )

    async def scenario():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            predict = asyncio.create_task(_predict(client))
            await asyncio.to_thread(started.wait, 5)
            ping = await client.get("/ping")
            answered_during_inference = not done.is_set()
            release.set()
            return ping, answered_during_inference, await predict

    ping, answered_during_inference, predict = asyncio.run(scenario())
    assert ping.text == "pong"
    assert answered_during_inference
    assert predict.status_code == 200


def test_human_faces_are_requested_while_animals_are_inferred(tmp_path, monkeypatch):
    serve, started, release, done = _slow_detector_serve(
        tmp_path, monkeypatch, UPSTREAM_ML_URL="http://upstream"
    )
    human = {"boundingBox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "score": 0.9}
    upstream_during_inference = []

    async def post_upstream(body, content_type):
        upstream_during_inference.append(not done.is_set())
        release.set()
        payload = orjson.dumps({"facial-recognition": [human]})
        return httpx.Response(200, content=payload, request=httpx.Request("POST", "/"))

    monkeypatch.setattr(serve, "_post_upstream", post_upstream)

    async def scenario():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            return await _predict(client)

    response = asyncio.run(scenario())
    assert upstream_during_inference == [True]
    assert response.json()["facial-recognition"] == [human]


def _upstream_serve(tmp_path, monkeypatch, handler):
    """serve whose upstream client answers through `handler`."""
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.zeros((1, 1, 6), np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    serve = _load_serve(tmp_path, monkeypatch, UPSTREAM_ML_URL="http://upstream")
    serve._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return serve


def test_upstream_requests_survive_dropped_connections(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) < 3:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return httpx.Response(200, json={"clip": "[0.1]"})

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    response = asyncio.run(serve._post_upstream(b"body", "multipart/form-data"))

    assert response.json() == {"clip": "[0.1]"}
    assert calls == ["/predict"] * 3


def test_ping_reports_an_unreachable_upstream(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    response = TestClient(serve.app).get("/ping")

    assert response.status_code == 503


def _plain_serve(tmp_path, monkeypatch, **env):
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.zeros((1, 1, 6), np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    return _load_serve(tmp_path, monkeypatch, **env)


def test_people_without_an_upstream_refuse_to_start(tmp_path, monkeypatch):
    # Animals alone would look like a full answer, and a face refresh in Immich
    # deletes the human faces it does not find again.
    monkeypatch.delenv("UPSTREAM_ML_URL", raising=False)
    try:
        _plain_serve(tmp_path, monkeypatch, KEEP_HUMAN_FACES="true")
    except SystemExit as error:
        assert "UPSTREAM_ML_URL" in str(error)
    else:
        raise AssertionError("serve started without an upstream")


def test_root_answers_while_upstream_is_down(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    client = TestClient(serve.app)

    assert client.get("/").status_code == 200
    assert client.get("/ping").status_code == 503


def test_embedder_batches_are_capped(tmp_path, monkeypatch):
    rows = [[i * 60, 10, i * 60 + 50, 60, 0.9, 0] for i in range(5)]
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    serve = _load_serve(
        tmp_path, monkeypatch, KEEP_HUMAN_FACES="false", EMBED_BATCH="2"
    )
    session = serve._models.embedders["embedding"].session
    sizes = []

    class Recording:
        def run(self, outputs, feeds):
            sizes.append(len(next(iter(feeds.values()))))
            return session.run(outputs, feeds)

    serve._models.embedders["embedding"] = serve._models.embedders[
        "embedding"
    ]._replace(session=Recording())
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )

    assert len(response.json()["facial-recognition"]) == 5
    assert sizes == [2, 2, 1]


def test_animal_inference_runs_one_photo_at_a_time(tmp_path, monkeypatch):
    serve = _plain_serve(tmp_path, monkeypatch, KEEP_HUMAN_FACES="false")
    running, peak, lock = [0], [0], threading.Lock()

    def detect(rgb):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        threading.Event().wait(0.1)
        with lock:
            running[0] -= 1
        return []

    monkeypatch.setattr(serve, "_detect", detect)

    async def scenario():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            return await asyncio.gather(*(_predict(client) for _ in range(3)))

    responses = asyncio.run(scenario())
    assert [r.status_code for r in responses] == [200] * 3
    assert peak[0] == 1


def test_upstream_connect_timeout_is_separate_from_read(tmp_path, monkeypatch):
    serve = _plain_serve(
        tmp_path,
        monkeypatch,
        UPSTREAM_ML_URL="http://upstream",
        UPSTREAM_TIMEOUT="300",
        UPSTREAM_CONNECT_TIMEOUT="2",
    )

    async def timeout():
        async with serve._lifespan(serve.app):
            return serve._client.timeout

    result = asyncio.run(timeout())
    assert (result.connect, result.read) == (2, 300)


def test_idle_models_are_unloaded_and_reloaded(tmp_path, monkeypatch):
    serve = _plain_serve(
        tmp_path, monkeypatch, KEEP_HUMAN_FACES="false", MODEL_TTL="60"
    )
    client = TestClient(serve.app)

    def predict():
        return client.post(
            "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
        )

    assert predict().status_code == 200
    assert not serve._unload_idle_models()  # used just now
    serve._models_last_used -= 61
    serve._acquire_models()
    assert not serve._unload_idle_models()  # a request still holds them
    serve._release_models()
    serve._models_last_used -= 61
    assert serve._unload_idle_models()
    assert serve._models is None

    assert predict().status_code == 200
    assert serve._models is not None


def test_faces_are_refused_while_upstream_is_down(tmp_path, monkeypatch):
    # Animals alone would look like a full answer to a face refresh.
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )

    assert response.status_code == 503
    assert "facial-recognition" not in response.text


def test_invalid_limits_refuse_to_start(tmp_path, monkeypatch):
    for env in ({"INFERENCE_CONCURRENCY": "0"}, {"EMBED_BATCH": "-1"}):
        try:
            _plain_serve(tmp_path, monkeypatch, KEEP_HUMAN_FACES="false", **env)
        except SystemExit as error:
            assert "at least 1" in str(error)
        else:
            raise AssertionError(f"serve started with {env}")
        for key in env:
            monkeypatch.delenv(key)


def test_lifespan_unloads_idle_models(tmp_path, monkeypatch):
    serve = _plain_serve(
        tmp_path, monkeypatch, KEEP_HUMAN_FACES="false", MODEL_TTL="0.1"
    )

    with TestClient(serve.app) as client:
        for _ in range(50):
            if serve._models is None:
                break
            threading.Event().wait(0.05)
        assert serve._models is None
        response = client.post(
            "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
        )
        assert response.status_code == 200


def test_upstream_answers_without_faces_are_refused(tmp_path, monkeypatch):
    bodies = iter([b'{"clip": "[0.1]"}', b"not json", b'{"facial-recognition": []}'])

    def handler(request):
        return httpx.Response(200, content=next(bodies))

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    client = TestClient(serve.app)

    def predict():
        return client.post(
            "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
        )

    assert [predict().status_code for _ in range(3)] == [503, 503, 200]


def test_passthrough_answers_503_while_upstream_is_down(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    response = TestClient(serve.app).post(
        "/predict",
        data={"entries": json.dumps({"clip": {}})},
        files={"image": _image()},
    )

    assert response.status_code == 503


def test_queued_inference_is_dropped_when_upstream_fails(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    serve = _upstream_serve(tmp_path, monkeypatch, handler)
    detected = []
    monkeypatch.setattr(serve, "_detect", lambda rgb: detected.append(1) or [])

    async def scenario():
        limiter = serve._inference_limiter()
        await limiter.acquire()  # another photo holds the only slot
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            response = await _predict(client)
        limiter.release()
        await asyncio.sleep(0.1)
        return response

    assert asyncio.run(scenario()).status_code == 503
    assert detected == []


def _face_log_serve(tmp_path, monkeypatch, handler, **env):
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.zeros((1, 1, 6), np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    serve = _load_serve(tmp_path, monkeypatch, UPSTREAM_ML_URL="http://upstream", **env)
    serve._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return serve


def _human(score: float) -> dict:
    box = {"x1": 1, "y1": 2, "x2": 30, "y2": 40}
    return {"boundingBox": box, "embedding": "[0.1]", "score": score}


def test_face_log_asks_lower_logs_all_and_answers_immichs_min_score(
    tmp_path, monkeypatch
):
    sent = []

    def handler(request):
        sent.append(request.content)
        faces = [_human(0.9), _human(0.55), _human(0.41)]
        return httpx.Response(200, json={"facial-recognition": faces})

    log_path = tmp_path / "faces.jsonl"
    serve = _face_log_serve(
        tmp_path,
        monkeypatch,
        handler,
        FACE_LOG_MIN_SCORE="0.4",
        FACE_LOG_PATH=str(log_path),
    )
    image = _image()
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": image}
    )

    assert [f["score"] for f in response.json()["facial-recognition"]] == [0.9]
    assert b'"minScore":0.4' in sent[0] and image in sent[0]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert record["minScore"] == 0.7
    assert [f["score"] for f in record["faces"]] == [0.9, 0.55, 0.41]
    assert record["sha1"] == hashlib.sha1(image).hexdigest()


def test_face_requests_are_untouched_without_face_log(tmp_path, monkeypatch):
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx.Response(200, json={"facial-recognition": [_human(0.5)]})

    monkeypatch.delenv("FACE_LOG_MIN_SCORE", raising=False)
    serve = _face_log_serve(tmp_path, monkeypatch, handler)
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )

    # Without the log the sidecar does not filter what upstream found.
    assert [f["score"] for f in response.json()["facial-recognition"]] == [0.5]
    assert b'"minScore": 0.7' in sent[0]


def test_face_log_forwards_unknown_entries_as_they_came(tmp_path, monkeypatch):
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx.Response(200, json={"facial-recognition": []})

    serve = _face_log_serve(tmp_path, monkeypatch, handler, FACE_LOG_MIN_SCORE="0.4")
    entries = {"facial-recognition": {"detection": {}}}
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(entries)}, files={"image": _image()}
    )

    assert response.status_code == 200
    assert b'"detection": {}' in sent[0]


def test_invalid_face_log_min_score_refuses_to_start(tmp_path, monkeypatch):
    try:
        _plain_serve(
            tmp_path, monkeypatch, KEEP_HUMAN_FACES="false", FACE_LOG_MIN_SCORE="1.5"
        )
    except SystemExit as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("serve started with FACE_LOG_MIN_SCORE=1.5")


def test_animal_log_keeps_weak_animals_with_their_embedding(tmp_path, monkeypatch):
    rows = [
        [10, 10, 100, 100, 0.50, 0],  # dog above DOG_MIN_SCORE: answered
        [200, 200, 300, 300, 0.20, 0],  # weak dog: logged with its embedding only
        [400, 400, 500, 500, 0.10, 0],  # below ANIMAL_LOG_MIN_SCORE: dropped
    ]
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    log_path = tmp_path / "faces.jsonl"
    serve = _load_serve(
        tmp_path,
        monkeypatch,
        KEEP_HUMAN_FACES="false",
        ANIMAL_LOG_MIN_SCORE="0.15",
        FACE_LOG_PATH=str(log_path),
    )

    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )

    answered = response.json()["facial-recognition"]
    assert [round(f["score"], 2) for f in answered] == [0.5]
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert "faces" not in record  # human logging is off
    animals = {round(a["score"], 2): a for a in record["animals"]}
    assert set(animals) == {0.5, 0.2}
    assert animals[0.5]["species"] == "dog" and "embedding" not in animals[0.5]
    assert len(json.loads(animals[0.2]["embedding"])) == 512


def test_animal_detection_is_unchanged_without_animal_log(tmp_path, monkeypatch):
    rows = [[10, 10, 100, 100, 0.50, 0], [200, 200, 300, 300, 0.20, 0]]
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))
    monkeypatch.delenv("ANIMAL_LOG_MIN_SCORE", raising=False)
    serve = _load_serve(tmp_path, monkeypatch, KEEP_HUMAN_FACES="false")

    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )

    assert [round(f["score"], 2) for f in response.json()["facial-recognition"]] == [
        0.5
    ]
    assert serve._face_log is None


PERSON_BOX = {"x1": 100, "y1": 100, "x2": 200, "y2": 220}


def _overlap_serve(tmp_path, monkeypatch, rows, clip_handler=None, **env):
    """serve with one human face upstream at PERSON_BOX and the given dog rows."""
    _constant_model(
        tmp_path / "detector.onnx",
        [1, 3, 640, 640],
        np.array([rows], np.float32),
        names="{0: 'dog'}",
    )
    _embedder(tmp_path, "embedding", np.ones(512, np.float32))

    def handler(request):
        if b"facial-recognition" in request.content:
            face = {"boundingBox": PERSON_BOX, "embedding": "[0.1]", "score": 0.9}
            return httpx.Response(200, json={"facial-recognition": [face]})
        return clip_handler(request)

    serve = _load_serve(tmp_path, monkeypatch, UPSTREAM_ML_URL="http://upstream", **env)
    serve._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return serve


def _answer_scores(serve) -> list[float]:
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )
    assert response.status_code == 200, response.text
    return sorted(round(f["score"], 2) for f in response.json()["facial-recognition"])


def _kinds_by_crop_width(serve, monkeypatch, kinds: dict[int, list[float]]):
    """CLIP stand-in: the crop width picks the [person, dog, cat] scores."""

    async def scores(crop):
        return np.array(kinds[crop.shape[1]], np.float32)

    monkeypatch.setattr(serve, "_overlap_scores", scores)


def test_animal_box_that_clip_calls_a_person_is_dropped(tmp_path, monkeypatch):
    rows = [
        [105, 105, 195, 215, 0.93, 0],  # a head taken for a dog, even at a high score
        [400, 400, 500, 500, 0.50, 0],  # a dog elsewhere: not checked, kept
    ]
    serve = _overlap_serve(tmp_path, monkeypatch, rows)
    # The padded dog box around the head is 108 px wide.
    _kinds_by_crop_width(serve, monkeypatch, {108: [0.10, 0.02, 0.03]})

    assert _answer_scores(serve) == [0.5, 0.9]


def test_pet_next_to_a_person_keeps_both_faces(tmp_path, monkeypatch):
    rows = [[0, 0, 640, 640, 0.93, 0]]  # a big dog box around the whole photo
    serve = _overlap_serve(tmp_path, monkeypatch, rows)
    # The dog crop looks like a dog; the face crop leans a little towards the dog.
    _kinds_by_crop_width(
        serve, monkeypatch, {640: [0.05, 0.12, 0.04], 100: [0.10, 0.12, 0.05]}
    )

    assert _answer_scores(serve) == [0.9, 0.93]


def test_human_face_that_clip_calls_a_dog_is_dropped(tmp_path, monkeypatch):
    rows = [[90, 90, 210, 230, 0.93, 0]]  # the dog's head, also found as a person
    serve = _overlap_serve(tmp_path, monkeypatch, rows)
    _kinds_by_crop_width(
        serve, monkeypatch, {144: [0.05, 0.14, 0.06], 100: [0.05, 0.15, 0.07]}
    )

    assert _answer_scores(serve) == [0.93]


def test_overlap_check_asks_upstream_clip_and_refuses_when_it_fails(
    tmp_path, monkeypatch
):
    seen, fail = [], [False]

    def clip_handler(request):
        if fail[0]:
            return httpx.Response(500)
        body = request.content
        # A text request has no file, so httpx sends it form-urlencoded, as the
        # CLIP test against the real immich-ml did.
        seen.append(b"text=" in body or b'name="text"' in body)
        if b'"textual"' in body:
            index = 0 if b"person" in body else 1 if b"dog" in body else 2
            vector = np.eye(3)[index].tolist()
        else:  # every crop looks like a person
            vector = [1.0, 0.0, 0.0]
        return httpx.Response(200, json={"clip": json.dumps(vector)})

    rows = [[105, 105, 195, 215, 0.93, 0]]
    serve = _overlap_serve(tmp_path, monkeypatch, rows, clip_handler)

    assert _answer_scores(serve) == [0.9]
    assert seen.count(True) == 6 and seen.count(False) == 1  # 6 prompts, 1 crop

    fail[0] = True
    serve._overlap_text = None
    response = TestClient(serve.app).post(
        "/predict", data={"entries": json.dumps(ENTRIES)}, files={"image": _image()}
    )
    assert response.status_code == 503


def test_overlapping_faces_stay_without_the_check(tmp_path, monkeypatch):
    rows = [[105, 105, 195, 215, 0.93, 0]]
    serve = _overlap_serve(tmp_path, monkeypatch, rows, OVERLAP_CHECK="false")

    assert _answer_scores(serve) == [0.9, 0.93]
