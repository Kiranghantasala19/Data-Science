from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# UI Page
@app.route("/")
def home():
    return render_template("index.html")


# Calculator API
@app.route("/calculate")
def calculate():

    try:
        a = float(request.args.get("a"))
        b = float(request.args.get("b"))
        op = request.args.get("op")

        if op == "add":
            result = a + b
        elif op == "sub":
            result = a - b
        elif op == "mul":
            result = a * b
        elif op == "div":
            result = a / b
        else:
            result = "Invalid"

        return jsonify({"result": result})

    except:
        return jsonify({"result": "Error"})


# Health API
@app.route("/health")
def health():
    return jsonify({"status": "API Working"})


# Image API
@app.route("/get")
def get_images():

    images = [
        "\static\img1.jpg",
        "\static\img2.jpg",
        "\static\img3.jpg",
        "\static\img4.jpg",
    
    ]

    return jsonify(images)


if __name__ == "__main__":
    app.run(debug=True)