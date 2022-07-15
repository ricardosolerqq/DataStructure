#  Diz respeito a cada dos tipos de conexão com cada uma das DATASOURCES, passando inclusive pelos métodos padrão para cada um delas.
class DataConnection:
    def __init__(self, name = None):
        self.name = name
        self.description = ""
        self.conectionString = ""
        self.key = ""
    def connect(self):
        pass
