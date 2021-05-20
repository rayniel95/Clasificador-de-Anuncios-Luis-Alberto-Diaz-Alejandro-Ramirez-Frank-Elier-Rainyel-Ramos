import joblib
from PyQt5.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QTextEdit,
                             QVBoxLayout, QWidget)

from classify import classify, classify_cnn, classify_rnn1, classify_rnn2

class MainWin(QWidget):
    def __init__(self, old_models, new_models):
        super().__init__()
        self.setWindowTitle('Clasificador de Anuncios')
        self._old_models = old_models
        self._new_models = new_models
        
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
        self._radiobtn2 = QRadioButton('knn model')
        self._radiobtn3 = QRadioButton('nb model')
        self._radiobtn4 = QRadioButton('svm model')
        self._radiobtn5 = QRadioButton('rf model')
        self._radiobtn6 = QRadioButton('cnn model')
        self._radiobtn7 = QRadioButton('rnn model')
        l2.addWidget(self._radiobtn1)
        l2.addWidget(self._radiobtn2)
        l2.addWidget(self._radiobtn3)
        l2.addWidget(self._radiobtn4)
        l2.addWidget(self._radiobtn5)
        l2.addWidget(self._radiobtn6)
        l2.addWidget(self._radiobtn7)
        self._radiobtn1.setChecked(True)

        l3 = QHBoxLayout()
        self._radiobtn8 = QRadioButton('dt model')
        self._radiobtn9 = QRadioButton('knn model')
        self._radiobtn10 = QRadioButton('nb model')
        self._radiobtn11 = QRadioButton('svm model')
        self._radiobtn12 = QRadioButton('rf model')
        self._radiobtn13 = QRadioButton('cnn model')
        self._radiobtn14 = QRadioButton('rnn model')
        l3.addWidget(self._radiobtn8)
        l3.addWidget(self._radiobtn9)
        l3.addWidget(self._radiobtn10)
        l3.addWidget(self._radiobtn11)
        l3.addWidget(self._radiobtn12)
        l3.addWidget(self._radiobtn13)
        l3.addWidget(self._radiobtn14)

        layout = ListLayout(self, [l1, l2, l3])
        
        self.resize(500, 100)

    def btn_on_click(self):
        if self._textbox1.toPlainText() == '':
            pass
        c = ''
        if self._radiobtn1.isChecked():
            c = classify(self._textbox1.toPlainText(), self._old_models[0][0], self._old_models[1][0], self._old_models[1][1])
            c = c[0]
        elif self._radiobtn2.isChecked():
            c = classify(self._textbox1.toPlainText(), self._old_models[0][1], self._old_models[1][0], self._old_models[1][1])
            c = c[0]
        elif self._radiobtn3.isChecked():
            c = classify(self._textbox1.toPlainText(), self._old_models[0][2], self._old_models[1][0], self._old_models[1][1])
            c = c[0]
        elif self._radiobtn4.isChecked():
            c = classify(self._textbox1.toPlainText(), self._old_models[0][3], self._old_models[1][0], self._old_models[1][1])
            c = c[0]
        elif self._radiobtn5.isChecked():
            c = classify(self._textbox1.toPlainText(), self._old_models[0][4], self._old_models[1][0], self._old_models[1][1])
            c = c[0]
        elif self._radiobtn6.isChecked():
            c = classify_cnn(self._old_models[0][5], self._textbox1.toPlainText(), self._old_models[1][2])
        elif self._radiobtn7.isChecked():
            c = classify_rnn1(self._old_models[0][6], self._textbox1.toPlainText())
            c = c[0]
        elif self._radiobtn8.isChecked():
            c = classify(self._textbox1.toPlainText(), self._new_models[0][0], self._new_models[1][0], self._new_models[1][1])
            c = c[0]
        elif self._radiobtn9.isChecked():
            c = classify(self._textbox1.toPlainText(), self._new_models[0][1], self._new_models[1][0], self._new_models[1][1])
            c = c[0]
        elif self._radiobtn10.isChecked():
            c = classify(self._textbox1.toPlainText(), self._new_models[0][2], self._new_models[1][0], self._new_models[1][1])
            c = c[0]
        elif self._radiobtn11.isChecked():
            c = classify(self._textbox1.toPlainText(), self._new_models[0][3], self._new_models[1][0], self._new_models[1][1])
            c = c[0]
        elif self._radiobtn12.isChecked():
            c = classify(self._textbox1.toPlainText(), self._new_models[0][4], self._new_models[1][0], self._new_models[1][1])
            c = c[0]
        elif self._radiobtn13.isChecked():
            c = classify_cnn(self._new_models[0][5], self._textbox1.toPlainText(), self._new_models[1][2], 1)
        elif self._radiobtn14.isChecked():
            c = classify_rnn2(self._new_models[0][6], self._textbox1.toPlainText(), self._new_models[1][2])
    
        self._textbox2.setText(c)

class ListLayout(QVBoxLayout):
    def __init__(self, QWidget, layouts):
        super().__init__(QWidget)
        for l in layouts:
            self.addLayout(l)

if __name__ == '__main__':

    o_dt = joblib.load('./old_models/dt')
    o_knn = joblib.load('./old_models/knn')
    o_nb = joblib.load('./old_models/nb')
    o_svm = joblib.load('./old_models/svm')
    o_rf = joblib.load('./old_models/rf')
    o_cnn = './old_models/cnn.hdf5'
    o_rnn_1 = joblib.load('./old_models/rnn')

    o_counter = joblib.load('./old_models/counter')
    o_tfidf_matrix = joblib.load('./old_models/tfidf_transformer')
    o_dp = './old_models/dataset_properties'

    n_dt = joblib.load('./new_models/n_dt')
    n_knn = joblib.load('./new_models/n_knn')
    n_nb = joblib.load('./new_models/n_nb')
    n_svm = joblib.load('./new_models/n_svm')
    n_rf = joblib.load('./new_models/n_rf')
    n_cnn = './new_models/n_cnn'
    n_rnn = './new_models/n_rnn'

    n_counter = joblib.load('./new_models/n_counter')
    n_tfidf_matrix = joblib.load('./new_models/n_tfidf_transformer')
    n_dp = './new_models/dataset_properties'

    app = QApplication([])
    win = MainWin(([o_dt, o_knn, o_nb, o_svm, o_rf, o_cnn, o_rnn_1], [o_counter, o_tfidf_matrix, o_dp]), ([n_dt, n_knn, n_nb, n_svm, n_rf, n_cnn, n_rnn], [n_counter, n_tfidf_matrix, n_dp]))
    win.show()
    app.exec()
