from flask import Flask, request, jsonify
from flask_cors import CORS
from beamframe.beam import *
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permitir cualquier origen (ajústalo si es necesario)

@app.route('/compute', methods=['POST'])
def compute():
    try:
        # Obtener datos del request
        body = request.json
        beam_length = body.get('beamLength')
        point_forces = body.get('pointForces', [])
        udls = body.get('udls', [])
        supports = body.get('supports', [])
        hinges = body.get('hinges', [])
        moments = body.get('pointMoments', [])

        # Validar y convertir la longitud de la viga
        beam_length = float(beam_length)

        # Crear la viga
        b = Beam(beam_length)

        # Agregar reacciones
        reactions = []
        for support in supports:
            reaction = Reaction(support['distance'], support['type'], support['label'])
            reactions.append(reaction)

        # Agregar cargas puntuales
        loads = []
        for pf in point_forces:
            load = PointLoad(float(pf['distance']), float(pf['magnitude']), inverted=True)
            loads.append(load)

        # Agregar momentos
        for moment in moments:
            load = PointMoment(float(moment['distance']), float(moment['magnitude']))
            loads.append(load)

        # Agregar cargas distribuidas
        for udl in udls:
            if udl['magnitudeStart'] == udl['magnitudeEnd']:
                udl_load = UDL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start']))
            else:
                udl_load = UVL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start']), float(udl['magnitudeEnd']))
            loads.append(udl_load)

        # Agregar bisagras
        for hinge in hinges:
            hinge_load = Hinge(float(hinge['distance']), hinge['side'])
            loads.append(hinge_load)

        # Resolver la viga y generar resultados
        b.fast_solve((*reactions, *loads))

        # Generar fuerzas cortantes y momentos flectores
        shear_forces = b.generate_shear_values((*reactions, *loads))
        bending_moments = b.generate_moment_values((*reactions, *loads))

        return jsonify({
            'status': 'success',
            'data': {
                'maxshearForces': max(shear_forces, default=0),
                'minshearForces': min(shear_forces, default=0),
                'bendingMoments': max(bending_moments, default=0),
                'minbendingMoments': min(bending_moments, default=0),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway asigna el puerto
    app.run(host='0.0.0.0', port=port)
