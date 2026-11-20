# coding:utf-8
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (
                                QHBoxLayout,
                                QLabel,
                                QVBoxLayout,
                                QWidget
                            )

from qfluentwidgets import (DoubleSpinBox, PushButton, ComboBox, FluentIcon)


class EditableDoubleSpinBox(DoubleSpinBox):
    """Double spin box whose number can be selected and edited.

    qfluentwidgets draws the up and down buttons with its own layout and then
    asks the style for the text field. With button symbols turned off, that
    field comes back 1 pixel wide, so the value is invisible and clicks land
    on the empty frame instead of the editor.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(140)
        edit = self.lineEdit()
        edit.setFrame(False)
        edit.setStyleSheet("background: transparent; border: none;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_edit_field()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_edit_field()

    def _place_edit_field(self):
        edit = self.lineEdit()
        if edit is None:
            return
        button_left = self.width()
        for button in (getattr(self, "upButton", None), getattr(self, "downButton", None)):
            if button is not None and button.isVisible() and button.x() > 8:
                button_left = min(button_left, button.x())
        if button_left >= self.width() or button_left <= 8:
            button_left = self.width() - 72
        edit.setGeometry(2, 1, max(24, button_left - 6), max(1, self.height() - 2))
        edit.raise_()
        for button in (getattr(self, "upButton", None), getattr(self, "downButton", None)):
            if button is not None:
                button.raise_()

from .navigationInterface import NavigationBaseInterface
from .fileNameListViewInterface import FileNameListView
from .outputLabelLineEditButtonWidget import OutputGroupWidget

from .style_sheet import StyleSheet

from .config import normalize_bandit_track_index


class BanditParamGroupWidget(QWidget):
    def __tr(self, text):
        return QCoreApplication.translate(self.__class__.__name__, text)

    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)

        self.vBoxLayout = QVBoxLayout()
        self.setLayout(self.vBoxLayout)
        self.mainWidget = QWidget()
        self.mainWidget.setObjectName("mainObject")

        self.mainLayout = QVBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.__tr_text_all_Stems = self.__tr("所有音轨")
        self.__tr_text_Speech = self.__tr("仅对白")
        self.__tr_text_Music = self.__tr("仅音乐")
        self.__tr_text_Effects = self.__tr("仅效果")
        self.__tr_text_Speech_Background = self.__tr("对白、背景二分")

        self.__tr_text_list = [
            self.__tr_text_all_Stems,
            self.__tr_text_Speech,
            self.__tr_text_Music,
            self.__tr_text_Effects,
            self.__tr_text_Speech_Background,
        ]

        self.setupUI()

    def addWidget(self, widget, alignment):
        self.mainLayout.addWidget(widget, alignment=alignment)

    def addLayout(self, layout):
        self.mainLayout.addLayout(layout)

    def setupUI(self):

        self.label_ = QLabel(self.__tr("BandIt Plus 参数"))
        self.vBoxLayout.addWidget(self.label_)

        self.vBoxLayout.addWidget(self.mainWidget)

        self.param_hBoxLayout = QHBoxLayout()
        self.addLayout(self.param_hBoxLayout)
        self.param_hBoxLayout.setContentsMargins(10, 10, 10, 10)

        self.vBoxLayout_overlap = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_overlap)

        self.spinBox_overlap = EditableDoubleSpinBox()
        self.spinBox_overlap.setMaximum(0.5)
        self.spinBox_overlap.setMinimum(0.0)
        self.spinBox_overlap.setSingleStep(0.05)
        self.spinBox_overlap.setValue(0.25)
        self.spinBox_overlap.setDecimals(2)
        self.spinBox_overlap.setToolTip(self.tr(
            "相邻分段的重叠比例。0.25 表示后一段与前一段重叠 25%，并用交叉淡化拼回去。比例越大，接缝越不明显，但处理更慢"
        ))

        self.label_overlap_param = QLabel(self.tr("采样重叠度"))

        self.vBoxLayout_overlap.addWidget(self.label_overlap_param)
        self.vBoxLayout_overlap.addWidget(self.spinBox_overlap)

        self.vBoxLayout_segment = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_segment)

        self.spinBox_segment = EditableDoubleSpinBox()
        self.spinBox_segment.setMaximum(100.0)
        self.spinBox_segment.setMinimum(1)
        self.spinBox_segment.setSingleStep(1)
        self.spinBox_segment.setValue(6.0)
        self.spinBox_segment.setDecimals(1)
        self.spinBox_segment.setToolTip(self.tr(
            "BandIt Plus 按该秒数分段推理。模型在 6 秒片段上训练。\n"
            "分段越长越占内存和显存。RTX 4070 12GB 上 6 到 10 秒比较合适"
        ))

        self.label_segment_param = QLabel(self.tr("分段长度(s)"))

        self.vBoxLayout_segment.addWidget(self.label_segment_param)
        self.vBoxLayout_segment.addWidget(self.spinBox_segment)

        self.vBoxLayout_stems = QVBoxLayout()
        self.param_hBoxLayout.addLayout(self.vBoxLayout_stems)

        self.comboBox_stems = ComboBox()
        for item in self.__tr_text_list:
            self.comboBox_stems.addItem(item)
        # Default matches the old "vocals only" choice: dialogue / speech.
        self.comboBox_stems.setCurrentIndex(1)

        self.label_stems_param = QLabel(self.tr("输出音轨"))

        self.vBoxLayout_stems.addWidget(self.label_stems_param)
        self.vBoxLayout_stems.addWidget(self.comboBox_stems)


class BanditPageNavigation(NavigationBaseInterface):
    def __init__(self, parent=None):
        super().__init__(
            title="BandIt Plus",
            subtitle=self.tr("使用 BandIt Plus 分离对白、音乐和效果。首次运行会把官方权重下载到 ./cache/dnr-3s-mus64-l1snr-plus.ckpt"),
            parent=parent,
        )

        self.setupUI()

        StyleSheet.BANDIT_INTERFACE.apply(self.bandit_param_widget)

    def setupUI(self):

        self.toolBar.modelStatusLabel.setVisible(False)

        self.fileListView = FileNameListView(self)
        self.addWidget(self.fileListView)

        self.bandit_param_widget = BanditParamGroupWidget(self)
        self.addWidget(self.bandit_param_widget)

        self.outputGroupWidget = OutputGroupWidget(self)
        self.addWidget(self.outputGroupWidget)

        self.process_button = PushButton()
        self.addWidget(self.process_button)
        self.process_button.setText(self.tr("提取"))
        self.process_button.setIcon(FluentIcon.IOT)

    def getParam(self) -> dict:
        param = {}
        param["overlap"] = self.bandit_param_widget.spinBox_overlap.value()
        param["segment"] = self.bandit_param_widget.spinBox_segment.value()
        param["tracks"] = self.bandit_param_widget.comboBox_stems.currentIndex()
        return param

    def setParam(self, param: dict) -> None:
        self.bandit_param_widget.spinBox_overlap.setValue(param["overlap"])
        self.bandit_param_widget.spinBox_segment.setValue(param["segment"])
        self.bandit_param_widget.comboBox_stems.setCurrentIndex(
            normalize_bandit_track_index(param.get("tracks", 1))
        )
