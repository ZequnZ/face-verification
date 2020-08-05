"""simple flask example"""
from flask import Flask, request

from evaluation import img_eva

app = Flask(__name__)


@app.route("/test", methods=["POST"])
def get_distance():
    data = request.get_json()
    print(data)
    path1 = data["path1"]
    path2 = data["path2"]
    print(("+").join([path1, path2]))
    return img_eva([path1, path2])


if __name__ == "__main__":
    app.run()
