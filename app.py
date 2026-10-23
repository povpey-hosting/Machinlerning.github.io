"""Flask app — Customer Segmentation (K-Means)
ដំណើរការ:  python app.py   →  បើក http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request
import segmentation as seg

app = Flask(__name__)


def parse_k(value, default=None):
    try:
        k = int(value)
    except (TypeError, ValueError):
        k = default or seg.METRICS["best_k"]
    return max(2, min(10, k))


@app.route("/")
def index():
    return render_template("index.html", best_k=seg.METRICS["best_k"])


@app.route("/api/analysis")
def api_analysis():
    return jsonify(seg.analyze(parse_k(request.args.get("k"))))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        income = float(data["income"])
        spending = float(data["spending"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "សូមបញ្ចូលលេខត្រឹមត្រូវ"}), 400
    return jsonify(seg.predict(parse_k(data.get("k")), income, spending))


if __name__ == "__main__":
    app.run(debug=True)
