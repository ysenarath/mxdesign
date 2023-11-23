from mxflow.config import config
from mxflow.server.model import Database

DATABASE_URL = config["server"]["database"]["url"]

db = Database(DATABASE_URL)
