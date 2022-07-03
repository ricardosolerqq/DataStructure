#  Diz respeito a cada dos agrupamentos de dados utilzados pela equipe.
class DataSource:
    def __init__(self, name = None):
        self.name = name
        self.description = ""
        self.connectionsList = []
