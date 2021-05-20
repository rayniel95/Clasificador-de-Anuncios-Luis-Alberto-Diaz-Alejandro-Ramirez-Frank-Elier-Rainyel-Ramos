import joblib
from PyQt5.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QTextEdit,
                             QVBoxLayout, QWidget)

from utils import classify

class MainWin(QWidget):
    def __init__(self, models, counter, tfidf_matrix):
        super().__init__()
        self.setWindowTitle('Clasificador de Anuncios')
        self._models = models
        self._counter = counter
        self._tfidf_matrix = tfidf_matrix
        
        l1 = QGridLayout()
        self._textbox1 = QTextEdit()
        pushbtn1 = QPushButton('->')
        pushbtn1.clicked.connect(self.btn_on_click)
        self._textbox2 = QLineEdit()
        self._textbox2.setReadOnly(True)
        l1.addWidget(QLabel('Anuncio'), 1, 1)
        l1.addWidget(QLabel('Categoria'), 1, 3)
        l1.addWidget(self._textbox1, 2, 1)
        l1.addWidget(pushbtn1, 2, 2)
        l1.addWidget(self._textbox2, 2, 3)

        l2 = QHBoxLayout()
        self._radiobtn1 = QRadioButton('dt model')
        self._radiobtn1.setChecked(True)
        self._radiobtn2 = QRadioButton('knn model')
        self._radiobtn3 = QRadioButton('nb model')
        self._radiobtn4 = QRadioButton('svm model')
        l2.addWidget(self._radiobtn1)
        l2.addWidget(self._radiobtn2)
        l2.addWidget(self._radiobtn3)
        l2.addWidget(self._radiobtn4)

        layout = ListLayout(self, [l1, l2])
        
        self.resize(500, 100)

    def btn_on_click(self):
        if self._textbox1.toPlainText() == '':
            pass
        c = ''
        if self._radiobtn1.isChecked():
            c = classify(self._textbox1.toPlainText(), self._models[0], self._counter, self._tfidf_matrix)
        elif self._radiobtn2.isChecked():
            c = classify(self._textbox1.toPlainText(), self._models[1], self._counter, self._tfidf_matrix)
        elif self._radiobtn3.isChecked():
            c = classify(self._textbox1.toPlainText(), self._models[2], self._counter, self._tfidf_matrix)
        elif self._radiobtn4.isChecked():
            c = classify(self._textbox1.toPlainText(), self._models[3], self._counter, self._tfidf_matrix)

        self._textbox2.setText(c[0])

class ListLayout(QVBoxLayout):
    def __init__(self, QWidget, layouts):
        super().__init__(QWidget)
        for l in layouts:
            self.addLayout(l)

if __name__ == '__main__':

    dt = joblib.load('./dt')
    knn = joblib.load('./knn')
    nb = joblib.load('./nb')
    svm = joblib.load('./svm')

    counter = joblib.load('./counter')
    tfidf_matrix = joblib.load('./tfidf_transformer')

    app = QApplication([])
    win = MainWin([dt, knn, nb, svm], counter, tfidf_matrix)
    win.show()
    app.exec()
