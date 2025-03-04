from flask import Flask, request, jsonify
from beamframe.beam import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permitir cualquier origen (ajústalo si es necesario)

@app.route('/compute', methods=['POST'])
def compute():
    try:
        # Obtener datos del request
        data = request.json
        beam_length = data.get('beamLength')
        point_forces = data.get('pointForces', [])
        udls = data.get('udls', [])
        supports = data.get('supports', [])
        hinges = data.get('hinges', [])
        moments = data.get('pointMoments', [])

        # Convertir beam_length a flotante
        beam_length = float(beam_length)

        # Crear la viga
        b = Beam(beam_length)

        # Agregar reacciones
        reactions = [Reaction(s['distance'], s['type'], s['label']) for s in supports]

        # Agregar cargas puntuales
        loads = [PointLoad(float(pf['distance']), float(pf['magnitude']), inverted=True) for pf in point_forces]

        # Agregar momentos
        loads.extend([PointMoment(float(m['distance']), float(m['magnitude'])) for m in moments])

        # Agregar cargas distribuidas
        for udl in udls:
            if udl['magnitudeStart'] == udl['magnitudeEnd']:
                loads.append(UDL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start'])))
            else:
                loads.append(UVL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start']), float(udl['magnitudeEnd'])))

        # Agregar bisagras
        loads.extend([Hinge(float(h['distance']), h['side']) for h in hinges])

        # Resolver la viga
        b.fast_solve((*reactions, *loads))

        # Generar resultados
        shear_forces = b.generate_shear_values((*reactions, *loads))
        bending_moments = b.generate_moment_values((*reactions, *loads))

        return jsonify({
            'status': 'success',
            'data': {
                'maxshearForces': max(shear_forces),
                'minshearForces': min(shear_forces),
                'bendingMoments': max(bending_moments),
                'minbendingMoments': min(bending_moments),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
