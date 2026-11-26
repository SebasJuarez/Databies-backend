"""Pruebas del backend de Databies (Flask + Structo desde PyPI)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402

E = 11e6  # kPa (C24, 11 kN/mm²)
I = 0.1 * 0.2**3 / 12  # m⁴ (100 × 200 mm)

SIMPLE = {
    "beamLength": 4,
    "supports": [
        {"distance": 0, "type": "h", "label": "Apoyo 1"},
        {"distance": 4, "type": "r", "label": "Apoyo 2"},
    ],
    "pointForces": [{"distance": 2, "magnitude": 10, "case": "live"}],
    "udls": [{"start": 0, "end": 4, "magnitudeStart": 2, "magnitudeEnd": 2, "case": "dead"}],
    "pointMoments": [],
    "hinges": [],
    "factors": {"dead": 1.35, "live": 1.5},
    "E": E,
    "I": I,
}


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def post(client, payload, url="/compute"):
    res = client.post(url, json=payload)
    return res.status_code, res.get_json()


def test_health(client):
    for url in ("/", "/health", "/structo/health"):
        data = client.get(url).get_json()
        assert data["status"] == "ok"
        assert data["structoVersion"]


def test_viga_simple_servicio_y_mayorada(client):
    code, body = post(client, SIMPLE)
    assert code == 200, body
    data = body["data"]
    assert data["classification"] == "ISOSTATIC"
    assert data["classificationLabel"] == "Isostática"

    serv, fact = data["bases"]["service"], data["bases"]["factored"]
    # Servicio: 10/2 + 2·4/2 = 9 kN en cada apoyo
    assert [r["Ry"] for r in serv["reactions"]] == pytest.approx([9, 9])
    # Mayorada: 1.5·10/2 + 1.35·2·4/2 = 12.9 kN
    assert [r["Ry"] for r in fact["reactions"]] == pytest.approx([12.9, 12.9])
    assert [r["label"] for r in fact["reactions"]] == ["Apoyo 1", "Apoyo 2"]
    assert fact["reactions"][1]["Mz"] is None

    # M máx mayorado = 15·4/4 + 2.7·16/8 = 20.4 kN·m en x = 2
    m = fact["moment"]["extremes"]["absMax"]
    assert m["value"] == pytest.approx(20.4, rel=1e-6)
    assert m["x"] == pytest.approx(2)
    assert data["maxBendingMoment"] == pytest.approx(20.4, rel=1e-6)
    assert fact["maxAbsShear"] == pytest.approx(12.9, rel=1e-6)

    # Deflexión de servicio (EI real): PL³/48EI + 5wL⁴/384EI, hacia abajo
    EI = E * I
    esperado = 10 * 4**3 / (48 * EI) + 5 * 2 * 4**4 / (384 * EI)
    y = serv["deflection"]
    assert y["realSection"] is True
    assert y["extremes"]["min"]["value"] == pytest.approx(-esperado, rel=1e-4)
    assert y["extremes"]["min"]["x"] == pytest.approx(2)
    assert "rotation" in serv
    assert data["maxDeflection"]["value"] == pytest.approx(-esperado, rel=1e-4)
    assert len(fact["shear"]["x"]) > 100
    # El aviso de Rx de Structo no aplica a la página
    assert not any("Rx" in w for w in data["warnings"])


def test_alias_structo(client):
    code, body = post(client, SIMPLE, "/structo/compute")
    assert code == 200
    assert body["data"]["solver"] == "structo"


def test_hiperestatica_empotrada_con_rodillo(client):
    payload = {
        **SIMPLE,
        "supports": [{"distance": 0, "type": "f"}, {"distance": 4, "type": "r"}],
        "pointForces": [],
        "udls": [{"start": 0, "end": 4, "magnitudeStart": 1, "magnitudeEnd": 1, "case": "dead"}],
        "factors": {"dead": 1, "live": 1},
    }
    code, body = post(client, payload)
    assert code == 200, body
    data = body["data"]
    assert data["classificationLabel"] == "Hiperestática de grado 1"
    reac = data["reactions"]
    # Empotrada–apoyada con w = 1: R_A = 5wL/8, R_B = 3wL/8, M_A = wL²/8
    assert reac[0]["Ry"] == pytest.approx(2.5, rel=1e-4)
    assert reac[1]["Ry"] == pytest.approx(1.5, rel=1e-4)
    assert abs(reac[0]["Mz"]) == pytest.approx(2, rel=1e-4)
    assert reac[0]["label"] == "Apoyo 1"


def test_voladizo_trapezoidal_y_momento(client):
    payload = {
        **SIMPLE,
        "supports": [{"distance": 0, "type": "f"}],
        "pointForces": [],
        "udls": [{"start": 0, "end": 2, "magnitudeStart": 1, "magnitudeEnd": 3, "case": "live"}],
        "pointMoments": [{"distance": 3, "magnitude": 4, "case": "dead"}],
        "factors": {"dead": 1, "live": 1},
    }
    code, body = post(client, payload)
    assert code == 200, body
    r = body["data"]["reactions"][0]
    assert r["Ry"] == pytest.approx(4)  # (1 + 3)/2 · 2
    assert r["Mz"] is not None


def test_articulacion_gerber(client):
    payload = {
        **SIMPLE,
        "beamLength": 8,
        "supports": [{"distance": 0, "type": "h"}, {"distance": 4, "type": "r"}, {"distance": 8, "type": "r"}],
        "hinges": [{"distance": 5, "side": "l"}],
        "pointForces": [],
        "udls": [{"start": 0, "end": 8, "magnitudeStart": 1, "case": "dead"}],
        "factors": {"dead": 1, "live": 1},
    }
    code, body = post(client, payload)
    assert code == 200, body
    assert sum(r["Ry"] for r in body["data"]["reactions"]) == pytest.approx(8)


def test_sin_e_i_avisa(client):
    payload = {k: v for k, v in SIMPLE.items() if k not in ("E", "I")}
    code, body = post(client, payload)
    assert code == 200
    assert body["data"]["bases"]["service"]["deflection"]["realSection"] is False
    assert any("EI = 1" in w for w in body["data"]["warnings"])


def test_factor_faltante_avisa(client):
    code, body = post(client, {**SIMPLE, "factors": {"dead": 1.35}})
    assert code == 200
    assert any("carga viva" in w for w in body["data"]["warnings"])


@pytest.mark.parametrize(
    "patch, mensaje",
    [
        ({"supports": [{"distance": 0, "type": "xx"}]}, "Tipo de apoyo no válido"),
        ({"supports": []}, "al menos un apoyo"),
        ({"supports": [{"distance": 0, "type": "r"}]}, "inestable"),
        ({"pointForces": [{"distance": "a", "magnitude": 1}]}, "no es un número"),
        ({"udls": [{"start": 1, "end": 1, "magnitudeStart": 2}]}, "inicio y fin iguales"),
        ({"beamLength": "no-es-numero"}, "no es un número"),
        ({"beamLength": 0}, "mayor que 0"),
        ({"hinges": [{"distance": 4}]}, "dentro"),
        ({"pointForces": [], "udls": []}, "no tiene cargas"),
        ({"factors": {"dead": -1, "live": 1.5}}, "mayor que 0"),
        ({"pointForces": [{"distance": 1, "magnitude": 1, "case": "viento"}]}, "'dead' o 'live'"),
        ({"pointForces": [{"distance": 1, "magnitude": 1}] * 61}, "máximo"),
    ],
)
def test_rechaza_entradas_invalidas(client, patch, mensaje):
    code, body = post(client, {**SIMPLE, **patch})
    assert code == 400
    assert body["status"] == "error"
    assert mensaje in body["error"]


def test_cuerpo_no_json(client):
    res = client.post("/compute", data="hola", content_type="text/plain")
    assert res.status_code == 400
