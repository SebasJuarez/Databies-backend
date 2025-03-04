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
        try:
            body = request.json
            beam_length = body.get('beamLength')
            point_forces = body.get('pointForces', [])
            udls = body.get('udls', [])
            supports = body.get('supports', [])
            hinges = body.get('hinges', [])
            moments = body.get('pointMoments', [])
        except Exception as e:
            return jsonify({'error': f"Error al procesar la solicitud: {str(e)}"}), 400

        # Validar y convertir la longitud de la viga
        try:
            beam_length = float(beam_length)
            b = Beam(beam_length)
        except Exception as e:
            return jsonify({'error': f"Error al inicializar la viga: {str(e)}"}), 400

        # Agregar reacciones
        reactions = []
        try:
            for support in supports:
                distance = float(support['distance'])
                support_type = support['type']
                label = support['label']
                reaction = Reaction(distance, support_type, label)
                reactions.append(reaction)
        except Exception as e:
            return jsonify({'error': f"Error al procesar soportes: {str(e)}"}), 400

        # Agregar cargas puntuales
        loads = []
        try:
            for pf in point_forces:
                load = PointLoad(float(pf['distance']), float(pf['magnitude']), inverted=True)
                loads.append(load)
        except Exception as e:
            return jsonify({'error': f"Error al procesar cargas puntuales: {str(e)}"}), 400

        # Agregar momentos
        try:
            for moment in moments:
                load = PointMoment(float(moment['distance']), float(moment['magnitude']))
                loads.append(load)
        except Exception as e:
            return jsonify({'error': f"Error al procesar momentos: {str(e)}"}), 400

        # Agregar cargas distribuidas
        try:
            for udl in udls:
                if udl['magnitudeStart'] == udl['magnitudeEnd']:
                    udl_load = UDL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start']))
                else:
                    udl_load = UVL(float(udl['start']), float(udl['magnitudeStart']), float(udl['end']) - float(udl['start']), float(udl['magnitudeEnd']))
                loads.append(udl_load)
        except Exception as e:
            return jsonify({'error': f"Error al procesar cargas distribuidas: {str(e)}"}), 400

        # Agregar bisagras
        try:
            for hinge in hinges:
                hinge_load = Hinge(float(hinge['distance']), hinge['side'])
                loads.append(hinge_load)
        except Exception as e:
            return jsonify({'error': f"Error al procesar bisagras: {str(e)}"}), 400

        # Resolver la viga y generar resultados
        try:
            b.fast_solve((*reactions, *loads))
        except Exception as e:
            return jsonify({'error': f"Error al resolver la viga: {str(e)}"}), 400

        # Generar fuerzas cortantes y momentos flectores
        try:
            shear_forces = b.generate_shear_values((*reactions, *loads))
            bending_moments = b.generate_moment_values((*reactions, *loads))
        except Exception as e:
            return jsonify({'error': f"Error al generar resultados: {str(e)}"}), 400

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
        return jsonify({'error': f"Error inesperado: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway asigna el puerto
    app.run(host='0.0.0.0', port=port)
