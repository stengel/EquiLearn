# imports
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import pdb
import game_solver
import os
import flask_assets

app = Flask(__name__)

app.config.update(TEMPLATES_AUTO_RELOAD = True)

env = flask_assets.Environment(app)

# Tell flask_assets where to look for our coffeescript and sass files.
env.load_path = [
    os.path.join(os.path.dirname(__file__), 'sass'),
    os.path.join(os.path.dirname(__file__), 'coffee'),
    os.path.join(os.path.dirname(__file__), 'bower_components')
]

env.register(
    'js_all',
    flask_assets.Bundle(
        'jquery/dist/jquery.min.js', 'jquery-ui/jquery-ui.min.js',
        flask_assets.Bundle(
            'all.coffee',
            filters=['coffeescript']
        ),
        output='js_all.js'
    )
)

env.register(
    'css_all',
    flask_assets.Bundle(
        'all.sass',
        filters='sass',
        output='css_all.css'
    )
)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        write_input_file(request.form)
        game_solver.run()

        with open('index_output', 'r') as file:
            data = json.loads(file.read())

        return jsonify(data)
    return render_template('index.html')

def write_input_file(form):
    matrix_a, matrix_b = form['A'], form['B']
    m, n = form['m'], form['n']
    with open('lrs/lrsnash_input', 'w') as file:
        file.write(m + ' ' + n + '\n\n')
        for row in json.loads(matrix_a):
            for item in row:
                file.write(item + " ")
            file.write('\n')

        file.write('\n')

        for row in json.loads(matrix_b):
            for item in row:
                file.write(item + " ")
            file.write('\n')

# run the server
if __name__ == '__main__':
    app.run(debug=True)
