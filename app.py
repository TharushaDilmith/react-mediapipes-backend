#import flask module
from flask import Flask
from resources.routes import initialize_routes
from flask_restful import Api

app = Flask(__name__)
api = Api(app)

#Test route
@app.route('/')
def hello_world():
    return 'Hello World'

initialize_routes(api)

#Main function
if __name__ == '__main__':
    app.run()
