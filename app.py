"""Backend de Databies: cálculo de vigas para la página de Diagramas.

Usa Structo, la librería del equipo publicada en PyPI como ``databies``
(``pip install databies``). La página manda la viga armada (apoyos, cargas con
su caso muerta/viva, factores de mayoración, E e I) y este servicio la resuelve
dos veces:

- ``service``: cargas nominales, para deflexión y rotación.
- ``factored``: cargas mayoradas, para V, M y reacciones de diseño (lo que usan
  los chequeos de Eurocódigo 5).

Rutas
-----
- ``POST /compute``          cálculo (también responde en ``/structo/compute``)
- ``GET  /health``           estado y versión de Structo (también ``/structo/health``)

Convenciones
------------
UI (Force.jsx)                      Structo
- Fuerza puntual: + hacia abajo     + hacia arriba   -> se invierte el signo
- Carga distribuida: + hacia abajo  + hacia arriba   -> se invierte el signo
- Momento: + antihorario            + antihorario    -> igual
- Reacciones: + hacia arriba        + hacia arriba   -> igual
- Momento flector: + tracción abajo (Structo ya lo da así)

Unidades: Structo no convierte nada. La UI trabaja en kN y m, así que los
resultados salen en kN, kN·m, rad y m. Para que la deflexión salga en metros,
``E`` debe venir en kPa (kN/m²) e ``I`` en m⁴. Si no vienen, se usa EI = 1 y la
deflexión solo sirve para ver la forma (reacciones, V y M no dependen de EI
porque la sección es constante).
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from databies import (
    Beam,
    CrossSection,
    DistributedLoad,
    FixedSupport,
    Material,
    MomentLoad,
    PinSupport,
    PointLoad,
    RollerSupport,
    analyze,
)
from databies import __version__ as STRUCTO_VERSION

app = Flask(__name__)
# Una viga con muchas cargas pesa pocos KB; esto corta peticiones absurdas.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# CORS_ORIGINS="https://databies.netlify.app,http://localhost:5173" para cerrarlo.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": _origins or "*"}})

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_SUPPORTS = {
    "h": ("pin", PinSupport),
    "pin": ("pin", PinSupport),
    "hinge": ("pin", PinSupport),
    "pasador": ("pin", PinSupport),
    "articulado": ("pin", PinSupport),
    "r": ("roller", RollerSupport),
    "roller": ("roller", RollerSupport),
    "rodillo": ("roller", RollerSupport),
    "f": ("fixed", FixedSupport),
    "fixed": ("fixed", FixedSupport),
    "empotrado": ("fixed", FixedSupport),
    "empotramiento": ("fixed", FixedSupport),
}

_CASES = {
    "dead": "dead",
    "d": "dead",
    "muerta": "dead",
    "cm": "dead",
    "live": "live",
    "l": "live",
    "viva": "live",
    "cv": "live",
}
_DEFAULT_CASE = "live"
_CASE_NAMES = {"dead": "carga muerta", "live": "carga viva"}
_BASIS_LABELS = {"nominal": "Servicio", "factored": "Mayorada"}

# Límites para no gastar el plan gratis de Render en peticiones raras.
MAX_SUPPORTS = 10
MAX_HINGES = 10
MAX_LOADS = 60
MAX_LENGTH = 1000.0  # m
SAMPLE_COUNT = 301  # puntos de cada diagrama
ELEMENTS_PER_SPAN = 8

# Decimales que se mandan al front (suficiente para graficar y mostrar).
_DECIMALS = 6

# Avisos de Structo que no aplican a la página (no hay cargas horizontales).
_IGNORED_WARNINGS = ("`Rx` no calculado",)


class PayloadError(ValueError):
    """Error en los datos que manda la página (no de Structo)."""


# ---------------------------------------------------------------------------
# Lectura del JSON
# ---------------------------------------------------------------------------


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PayloadError(f"`{name}` no es un número válido: {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"`{name}` no es un número válido: {value!r}") from exc
    if not math.isfinite(out):
        raise PayloadError(f"`{name}` debe ser un número finito.")
    return out


def _list(payload: dict, key: str, limit: int) -> list[dict]:
    items = payload.get(key) or []
    if not isinstance(items, list):
        raise PayloadError(f"`{key}` debe ser una lista.")
    if len(items) > limit:
        raise PayloadError(f"Demasiados elementos en `{key}` (máximo {limit}).")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise PayloadError(f"`{key}[{i}]` debe ser un objeto.")
    return items


def _case(item: dict, name: str) -> str:
    raw = item.get("case")
    if raw in (None, ""):
        return _DEFAULT_CASE
    key = str(raw).strip().lower()
    if key not in _CASES:
        raise PayloadError(f"`{name}.case` debe ser 'dead' o 'live'; se recibió {raw!r}.")
    return _CASES[key]


def _factors(payload: dict) -> dict[str, float]:
    """Factores de mayoración por caso; 1.0 si no vienen."""

    raw = payload.get("factors") or {}
    if not isinstance(raw, dict):
        raise PayloadError("`factors` debe ser un objeto como {\"dead\": 1.35, \"live\": 1.5}.")
    out: dict[str, float] = {}
    for case in ("dead", "live"):
        value = raw.get(case)
        if value in (None, ""):
            out[case] = 1.0
            continue
        factor = _num(value, f"factors.{case}")
        if factor <= 0:
            raise PayloadError(f"El factor de mayoración de {_CASE_NAMES[case]} debe ser mayor que 0.")
        out[case] = factor
    return out


def _linear_load(w_start: float, w_end: float, a: float, b: float):
    """Intensidad lineal entre (a, w_start) y (b, w_end), ya en convención Structo."""

    slope = (w_end - w_start) / (b - a)

    def func(x: float) -> float:
        return -(w_start + slope * (x - a))

    return func


def build_beam(payload: dict) -> tuple[Beam, list[dict], list[str], dict[str, float], bool]:
    """Traduce el JSON de la página a un ``Beam`` de Structo.

    Returns:
        ``(beam, supports_meta, notes, factors, real_section)``. ``supports_meta``
        guarda label/tipo de cada apoyo en el orden de la página; ``notes``
        explica qué se asumió.
    """

    notes: list[str] = []
    length = _num(payload.get("beamLength"), "beamLength")
    if length <= 0:
        raise PayloadError("La longitud de la viga debe ser mayor que 0.")
    if length > MAX_LENGTH:
        raise PayloadError(f"La viga no puede medir más de {MAX_LENGTH:g} m.")
    factors = _factors(payload)

    E = payload.get("E")
    I = payload.get("I")  # noqa: E741
    real_section = E not in (None, "", 0) and I not in (None, "", 0)
    if real_section:
        e_val, i_val = _num(E, "E"), _num(I, "I")
        if e_val <= 0 or i_val <= 0:
            raise PayloadError("E e I deben ser mayores que 0.")
        material, section = Material(E=e_val), CrossSection(I=i_val)
    else:
        material, section = Material(E=1.0), CrossSection(I=1.0)
        notes.append(
            "Sin E e I de la sección: se usó EI = 1. Reacciones, V y M son "
            "correctos; la deflexión solo muestra la forma."
        )

    beam = Beam(length=length, material=material, section=section)

    # 1) Apoyos (primero, porque las articulaciones se validan contra ellos)
    supports = _list(payload, "supports", MAX_SUPPORTS)
    if not supports:
        raise PayloadError("Agrega al menos un apoyo.")
    supports_meta: list[dict] = []
    for i, s in enumerate(supports):
        key = str(s.get("type", "")).strip().lower()
        if key not in _SUPPORTS:
            raise PayloadError(f"Tipo de apoyo no válido: {s.get('type')!r}")
        kind, cls = _SUPPORTS[key]
        support = cls(position=_num(s.get("distance"), f"supports[{i}].distance"))
        beam.add_support(support)
        supports_meta.append(
            {
                "label": (str(s.get("label") or "").strip() or f"Apoyo {i + 1}"),
                "type": kind,
                "position": support.position,
                "obj": support,
            }
        )

    # 2) Articulaciones internas
    for i, h in enumerate(_list(payload, "hinges", MAX_HINGES)):
        beam.add_hinge(_num(h.get("distance"), f"hinges[{i}].distance"))

    point_forces = _list(payload, "pointForces", MAX_LOADS)
    moments = _list(payload, "pointMoments", MAX_LOADS)
    udls = _list(payload, "udls", MAX_LOADS)
    if len(point_forces) + len(moments) + len(udls) > MAX_LOADS:
        raise PayloadError(f"Demasiadas cargas (máximo {MAX_LOADS}).")

    # 3) Fuerzas puntuales (UI: + abajo -> Structo: + arriba)
    for i, pf in enumerate(point_forces):
        mag = _num(pf.get("magnitude"), f"pointForces[{i}].magnitude")
        pos = _num(pf.get("distance"), f"pointForces[{i}].distance")
        if mag == 0:
            continue
        case = _case(pf, f"pointForces[{i}]")
        beam.add_load(PointLoad(position=pos, magnitude=-mag, case=case, factor=factors[case]))

    # 4) Momentos (antihorario + en ambos lados)
    for i, m in enumerate(moments):
        mag = _num(m.get("magnitude"), f"pointMoments[{i}].magnitude")
        pos = _num(m.get("distance"), f"pointMoments[{i}].distance")
        if mag == 0:
            continue
        case = _case(m, f"pointMoments[{i}]")
        beam.add_load(MomentLoad(position=pos, magnitude=mag, case=case, factor=factors[case]))

    # 5) Cargas distribuidas: rectangulares, triangulares y trapezoidales
    for i, u in enumerate(udls):
        w1 = _num(u.get("magnitudeStart"), f"udls[{i}].magnitudeStart")
        w2 = _num(u.get("magnitudeEnd", w1), f"udls[{i}].magnitudeEnd")
        a = _num(u.get("start"), f"udls[{i}].start")
        b = _num(u.get("end"), f"udls[{i}].end")
        if w1 == 0 and w2 == 0:
            continue
        if b < a:  # igual que el dibujo: si vienen al revés se voltean
            a, b, w1, w2 = b, a, w2, w1
        if a == b:
            raise PayloadError(f"La carga distribuida #{i + 1} tiene inicio y fin iguales ({a:g} m).")
        case = _case(u, f"udls[{i}]")
        beam.add_load(
            DistributedLoad(start=a, end=b, func=_linear_load(w1, w2, a, b), case=case, factor=factors[case])
        )

    if not beam.loads:
        raise PayloadError("La viga no tiene cargas: agrega al menos una carga distinta de 0.")

    # Aviso de factores asumidos, solo para los casos que sí tienen cargas
    raw = payload.get("factors") or {}
    for case in sorted({load.case for load in beam.loads}, key=["dead", "live"].index):
        if raw.get(case) in (None, ""):
            notes.append(
                f"Sin factor de mayoración para {_CASE_NAMES[case]}: se usó 1.0 "
                "(la combinación mayorada es igual a la de servicio para ese caso)."
            )

    return beam, supports_meta, notes, factors, real_section


# ---------------------------------------------------------------------------
# Respuesta
# ---------------------------------------------------------------------------


def _r(v: float) -> float:
    out = round(float(v), _DECIMALS)
    return 0.0 if out == 0 else out  # sin -0.0


def _to_list(values: np.ndarray) -> list[float]:
    return [_r(v) for v in values]


def _extreme(x: np.ndarray, *series: np.ndarray) -> dict[str, dict[str, float]]:
    """Máximo, mínimo y máximo absoluto con su posición, revisando ambos lados de cada salto."""

    xs = np.concatenate([x for _ in series])
    vs = np.concatenate(series)
    i_max, i_min = int(np.argmax(vs)), int(np.argmin(vs))
    i_abs = i_max if abs(vs[i_max]) >= abs(vs[i_min]) else i_min
    return {
        "max": {"x": _r(xs[i_max]), "value": _r(vs[i_max])},
        "min": {"x": _r(xs[i_min]), "value": _r(vs[i_min])},
        "absMax": {"x": _r(xs[i_abs]), "value": _r(vs[i_abs])},
    }


def _classification(code: str | None) -> dict[str, Any]:
    code = code or ""
    if code == "ISOSTATIC":
        return {"code": code, "degree": 0, "label": "Isostática"}
    if code.startswith("HYPERSTATIC"):
        try:
            degree = int(code[code.index("(") + 1 : code.index(")")])
        except ValueError:
            degree = None
        label = "Hiperestática" + (f" de grado {degree}" if degree else "")
        return {"code": code, "degree": degree, "label": label}
    return {"code": code, "degree": None, "label": code}


def _block(results, supports_meta: list[dict], real_section: bool) -> dict[str, Any]:
    """Serializa un ``Results`` de Structo para la página."""

    reactions = []
    for meta in supports_meta:
        comps = results.reactions.get(meta["obj"], {})
        reactions.append(
            {
                "label": meta["label"],
                "type": meta["type"],
                "position": meta["position"],
                # None = ese componente no aplica (rodillo sin M, etc.)
                "Rx": _r(comps["Rx"]) if "Rx" in comps else None,
                "Ry": _r(comps["Ry"]) if "Ry" in comps else None,
                "Mz": _r(comps["Mz"]) if "Mz" in comps else None,
            }
        )

    shear, moment = results.shear, results.moment
    v_ext = _extreme(shear.x, shear.V_left, shear.V_right)
    m_ext = _extreme(moment.x, moment.M_left, moment.M_right)
    block: dict[str, Any] = {
        "loadBasis": results.load_basis,
        "label": _BASIS_LABELS.get(results.load_basis, results.load_basis),
        "loadCases": list(results.load_cases or ()),
        "reactions": reactions,
        "shear": {
            "x": _to_list(shear.x),
            "left": _to_list(shear.V_left),
            "right": _to_list(shear.V_right),
            "discontinuities": _to_list(shear.discontinuities),
            "extremes": v_ext,
        },
        "moment": {
            "x": _to_list(moment.x),
            "left": _to_list(moment.M_left),
            "right": _to_list(moment.M_right),
            "discontinuities": _to_list(moment.discontinuities),
            "extremes": m_ext,
        },
        # Nombres de siempre (los usan los chequeos de la página)
        "maxShearForce": v_ext["max"]["value"],
        "minShearForces": v_ext["min"]["value"],
        "maxBendingMoment": m_ext["max"]["value"],
        "minBendingMoment": m_ext["min"]["value"],
        "maxAbsShear": abs(v_ext["absMax"]["value"]),
        "maxAbsMoment": abs(m_ext["absMax"]["value"]),
    }
    if results.deflection is not None:
        d = results.deflection
        block["deflection"] = {
            "x": _to_list(d.x),
            # La deflexión en m es chica: se manda con más precisión.
            "y": [float(f"{v:.6e}") for v in d.y],
            "extremes": _extreme(d.x, np.asarray(d.y)),
            "realSection": real_section,
        }
    if results.rotation is not None:
        rot = results.rotation
        block["rotation"] = {
            "x": _to_list(rot.x),
            "left": [float(f"{v:.6e}") for v in rot.theta_left],
            "right": [float(f"{v:.6e}") for v in rot.theta_right],
            "discontinuities": _to_list(rot.discontinuities),
            "extremes": _extreme(rot.x, np.asarray(rot.theta_left), np.asarray(rot.theta_right)),
            "realSection": real_section,
        }
    return block


def solve_payload(payload: dict) -> dict:
    """Arma la viga, la resuelve con Structo (servicio y mayorada) y devuelve un dict serializable."""

    beam, supports_meta, notes, factors, real_section = build_beam(payload)
    try:
        per_span = int(payload.get("elementsPerSpan") or ELEMENTS_PER_SPAN)
    except (TypeError, ValueError):
        per_span = ELEMENTS_PER_SPAN
    options = {"sample_count": SAMPLE_COUNT, "elements_per_span": min(max(per_span, 2), 40)}

    service = analyze(beam, **options)
    factored = analyze(beam, factored=True, **options)

    warnings: list[str] = list(notes)
    for w in (*service.warnings, *factored.warnings):
        if w not in warnings and not w.startswith(_IGNORED_WARNINGS):
            warnings.append(w)

    bases = {
        "service": _block(service, supports_meta, real_section),
        "factored": _block(factored, supports_meta, real_section),
    }
    design, serv = bases["factored"], bases["service"]
    info = _classification(factored.classification)

    data: dict[str, Any] = {
        "solver": "structo",
        "structoVersion": STRUCTO_VERSION,
        "classification": info["code"],
        "classificationLabel": info["label"],
        "hyperstaticDegree": info["degree"],
        "warnings": warnings,
        "factors": factors,
        "loadCases": list(factored.load_cases or ()),
        "bases": bases,
        # Atajos de diseño (mayorada). Los diagramas completos van solo en `bases`
        # para no mandar los mismos arreglos dos veces.
        "reactions": design["reactions"],
        "maxShearForce": design["maxShearForce"],
        "minShearForces": design["minShearForces"],
        "maxBendingMoment": design["maxBendingMoment"],
        "minBendingMoment": design["minBendingMoment"],
    }
    if "deflection" in serv:
        data["maxDeflection"] = serv["deflection"]["extremes"]["absMax"]
    return data


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------


@app.get("/")
@app.get("/health")
@app.get("/structo/health")
def health():
    # La página la llama al abrir Diagramas para ir despertando Render.
    return jsonify({"status": "ok", "solver": "structo", "structoVersion": STRUCTO_VERSION})


@app.post("/compute")
@app.post("/structo/compute")
def compute():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "error": "El cuerpo debe ser un JSON."}), 400
    try:
        return jsonify({"status": "success", "data": solve_payload(payload)})
    except (ValueError, TypeError) as exc:  # PayloadError hereda de ValueError
        # Structo levanta ValueError con mensajes en español (viga inestable,
        # apoyo duplicado, articulación sobre un apoyo, etc.): se pasan tal cual.
        return jsonify({"status": "error", "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Error inesperado al resolver la viga")
        return jsonify({"status": "error", "error": f"Error inesperado en Structo: {exc}"}), 500


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"status": "error", "error": "La petición es demasiado grande."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
