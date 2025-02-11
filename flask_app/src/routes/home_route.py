from flask import Blueprint, jsonify, send_file, request, render_template
import io
import time

from src.utils.hair_color import convert

home_blueprint = Blueprint("home", __name__, template_folder="templates")


@home_blueprint.route("/")
def index():
    return render_template("index.html")


@home_blueprint.route("/api/health")
def health():
    return jsonify({"status": "ok"}), 200


@home_blueprint.route("/api/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    color_code = request.form["color"]

    output_image = convert(file, color_code)

    img_io = io.BytesIO()
    output_image.save(img_io, format="JPEG")
    img_io.seek(0)

    time.sleep(2.5)

    print(color_code)
    return send_file(
        img_io,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="output_image.jpg",
    )
