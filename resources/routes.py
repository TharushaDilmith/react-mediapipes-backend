
from .setData import DataSet
from .trainModel import TrainData
from .handIdentification import HandIdentifier

def initialize_routes(api):
    # api.add_resource(FeatureDetaction, "/api/face-detect")
    api.add_resource(DataSet, "/api/create-dataset")
    api.add_resource(TrainData, "/api/train-model")
    # api.add_resource(HandIdentifier, "/api/pose-identifier")

    # post api
    api.add_resource(HandIdentifier, "/api/pose-identifier", methods=['POST'])