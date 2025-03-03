from flask import Flask, render_template, redirect, url_for, request
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS
from flask_caching import Cache
from main import further_classifier

app = Flask(__name__)

CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_TIMEOUT':1800})

handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        output_bytes = subprocess.check_output(['python', 'main.py'], stderr=subprocess.STDOUT)
        output_str = output_bytes.decode('utf-8').strip()
        class_name, image_path = output_str.split(",")
        class_index = int(class_name)
        cache.set('class_index', class_index)
        cache.set('image_path', image_path)
        return redirect(url_for('output', class_index=class_index))
    except subprocess.CalledProcessError as e:
        error_message = f"Error executing main.py: {e}"
        return render_template('error.html', error_message=error_message)


@app.route('/output/<int:class_index>')
def output(class_index):
    rendered_template = render_template('output.html', output=class_index)
    return rendered_template


@app.route('/run_trans_app')
def run_trans_app():
    class_index = cache.get('class_index')
    image_path = cache.get('image_path')
    class_index = int(class_index)
    if class_index is not None and image_path is not None:
        further_output = further_classifier(r"{}".format(image_path), class_index)
        further_output_cleaned = further_output.strip()
        return redirect(url_for('further_output_page', further_output=further_output_cleaned))
    else:
        return render_template('error.html', error_message="Cache is empty")


@app.route('/run_cap_app')
def run_cap_app():
    class_index = cache.get('class_index')
    image_path = cache.get('image_path')
    class_index = int(class_index)
    if class_index is not None and image_path is not None:
        further_output = further_classifier(r"{}".format(image_path), class_index)
        further_output_cleaned = further_output.strip()
        return redirect(url_for('further_output_page', further_output=further_output_cleaned))
    else:
        return render_template('error.html', error_message="Cache is empty")


@app.route('/further_classf/<further_output>')
def further_output_page(further_output):
    rendered_template = render_template('further.html', output=further_output)
    return rendered_template


from flask import redirect, url_for

@app.route('/f_output', methods=['GET'])
def f_output():
    component = request.args.get('component')
    if component == 'cap':
        return redirect(url_for('run_cap_app'))
    elif component == 'trans':
        return redirect(url_for('run_trans_app'))
    else:
        further_output = "Unknown component"
        return render_template('further.html', output=further_output)


@app.route('/error')
def error():
    error_message = "An error occurred. Please try again later."
    return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run(host='localhost', debug=True, use_reloader=False)
