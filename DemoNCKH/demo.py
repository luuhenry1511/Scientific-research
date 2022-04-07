from flask import Flask, render_template, request, url_for
import pickle
import subprocess
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('nckh.html')
@app.route('/diemdanh')
def diemdanh():
    subprocess.call("Nhandien.py", shell=True)
    return "Đã hoàn tất quá trình điểm danh. Hãy quay lại trang chủ"

@app.route('/dulieutho')
def dulieutho():
    return render_template("dulieutho.html")

@app.route('/trangchu')
def trangchu():
    return render_template('nckh.html')

@app.route('/phantich')
def phantich():
    return "Chưa hoàn thiện"
if __name__ == '__main__':
    app.run()