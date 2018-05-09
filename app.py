# -*- coding: utf-8 -*-
from flask import Flask, request, render_template

import summarizer_machovec_modified

app = Flask(__name__)

app.config['DEBUG'] = True


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    text = request.form.get('text') or ''
    summary = ''
    if text:
        print('Summarizing...')
        summary = summarizer_machovec_modified.summarize(text)
    print(f'\n======summary======\n{summary}')
    return render_template('index.html', text=text, summary=summary)


if __name__ == '__main__':
    app.run()
