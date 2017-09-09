import os
from io import StringIO

from ansi2html import Ansi2HTMLConverter

from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import ocimatic
from ocimatic import filesystem, core, ui
from ocimatic.filesystem import Directory, FilePath
from importlib.util import find_spec


if find_spec('ansi2html') is not None:
    def ansi2html(ansi):
        return Ansi2HTMLConverter().convert(ansi)
else:
    def ansi2html(ansi):
        return ansi

UPLOAD_FOLDER = '/tmp/ocimatic/server/'

app = Flask(__name__)
app.secret_key = 'M\xf2\xba\xc0\xe3\xe55\xa0"\xff\x96\xba\xb8Jn\xc6#S\xa0t\xda\xb5[\r'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_file(uploaded_file):
    dst_dir = filesystem.Directory(UPLOAD_FOLDER, create=True)
    filename = secure_filename(uploaded_file.filename)
    filepath = FilePath(dst_dir, filename)
    uploaded_file.save(str(FilePath(dst_dir, filename)))
    return filepath


@app.route('/', methods=['POST', 'GET'])
def server():
    global contest
    result = ''
    if request.method == 'POST':
        uploaded_file = request.files.get('solution')
        if not uploaded_file or uploaded_file.filename == '':
            flash('Please select a file')
            return redirect(request.url)
        filepath = save_file(uploaded_file)
        task = contest.find_task(request.form.get('task'))
        solution = task.get_solution(filepath)
        if solution:
            stream = StringIO()
            with ui.capture_io(stream) as output:
                task.run_solution(solution)
            result = stream.getvalue()
        else:
            result = 'Invalid file'
    result = ansi2html(result)
    return render_template('index.html', tasks=contest.tasks, result=result)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    return render_template('submit.html')


def run(port=9999):
    global contest
    contest_dir = filesystem.change_directory()[0]
    contest = core.Contest(contest_dir)
    ocimatic.config['verbosity'] = 1
    app.run(port=port, debug=True)
