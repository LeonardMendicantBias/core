from flask import Blueprint
from controllers import Tasks


blueprint = Blueprint('blueprint', __name__)

blueprint.route('/', methods=['GET'])(Tasks.index)